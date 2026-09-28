"""Executa o gerador real com uma análise salva. Nenhuma API de pesquisa."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import html
import json
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.output.explanation_context import restore_result
from pipeline.output.explanation_generator import ExplanationGenerator


def load_analysis(analysis_id):
    import psycopg2
    url = os.getenv("HIBRIA_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("URL do banco não configurada.")
    connection = psycopg2.connect(url, connect_timeout=10)
    try:
        connection.set_session(readonly=True)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '30s'")
            cursor.execute("SELECT resultado_json, conteudo FROM analises WHERE id=%s", (analysis_id,))
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Análise não encontrada.")
        return row[0], row[1]
    finally:
        connection.rollback()
        connection.close()


def save_explanation(analysis_id, original, report):
    """Opt-in: apenas redação/metadados, com proteção contra atualização concorrente."""
    import psycopg2
    from psycopg2.extras import Json
    if report["source"] != "qwen":
        raise RuntimeError("Gravação recusada: a resposta não foi integralmente aceita. Veja a prévia.")
    payload = copy.deepcopy(original)
    metadata = payload.setdefault("metadata", {})
    history = metadata.get("explanation_history", [])
    if not isinstance(history, list):
        history = []
    metadata["explanation_history"] = (history + [{
        "explanation": original.get("explanation"), "details": original.get("details"),
        "source": metadata.get("explanation_source"),
        "version": metadata.get("explanation_version"),
        "replaced_at": datetime.now(timezone.utc).isoformat(),
    }])[-5:]
    payload.update(explanation=report["explanation"], details=report["details"])
    metadata.update(explanation_source=report["source"], explanation_version=report["version"],
                    explanation_validation=report.get("validation", {}))
    url = os.getenv("HIBRIA_DATABASE_URL") or os.getenv("DATABASE_URL")
    connection = psycopg2.connect(url, connect_timeout=10)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '30s'")
                cursor.execute("""
                    UPDATE analises SET explicacao=%s, resultado_json=%s
                    WHERE id=%s AND resultado_json=%s
                """, (report["explanation"], Json(payload), analysis_id, Json(original)))
                if cursor.rowcount != 1:
                    raise RuntimeError("A análise mudou durante o teste. Nada foi gravado; execute novamente.")
    finally:
        connection.close()


def make_html(title, old, final):
    def panel(heading, report):
        return ("<section><h2>" + html.escape(heading) + "</h2><h3>Evidências</h3><p>" +
                html.escape(str(report.get("explanation") or "")) + "</p><h3>Detalhes</h3><ul>" +
                "".join("<li>" + html.escape(str(item)) + "</li>" for item in report.get("details") or []) + "</ul></section>")
    return ("<!doctype html><html lang='pt-BR'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Comparação da explicação</title><style>body{font:17px/1.6 system-ui;background:#eef2f7;color:#142339;margin:30px auto;max-width:1100px;padding:0 18px}"
            "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px}section{padding:24px;background:white;border-radius:16px}h1{font-size:23px}h2{color:#1256a0}li{margin-bottom:14px}</style>"
            "<h1>" + html.escape(title) + "</h1><p>Versão " + ExplanationGenerator.VERSION +
            " · origem: " + html.escape(final["source"]) + "</p><main>" +
            panel("Salva anteriormente", old) + panel("Nova redação", final) + "</main></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analysis-id")
    group.add_argument("--input", type=Path, help="JSON exportado ou resultado_json")
    parser.add_argument("--env-file", default="/home/ubuntu/.config/hibria/hibria.env")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/explanation_v6"))
    parser.add_argument("--no-model", action="store_true", help="Inspeciona contexto e fallback sem Ollama")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--save-explanation", action="store_true", help="Grava só a redação, após aceitação integral")
    args = parser.parse_args()
    if args.save_explanation and (not args.analysis_id or args.no_model):
        parser.error("--save-explanation exige --analysis-id e uma chamada ao modelo.")
    load_dotenv(args.env_file, override=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.input:
        document = json.loads(args.input.read_text(encoding="utf-8-sig"))
        saved = document.get("analise")
        payload = saved["resultado_json"] if saved else document
        content = saved.get("conteudo", "") if saved else ""
    else:
        payload, content = load_analysis(args.analysis_id)
    result = restore_result(payload, content)
    trace = {}
    if args.no_model:
        context = ExplanationGenerator._build_context(result)
        request, sent, diagnostics = ExplanationGenerator._request_payload(result, context)
        final = ExplanationGenerator.fallback_report(result)
        trace.update(full_context=context, sent_context=sent, input_diagnostics=diagnostics,
                     request_preview=request, final=final,
                     mode="offline_sem_modelo")
    else:
        final = ExplanationGenerator.generate(result, trace=trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path, html_path = args.output.with_suffix(".json"), args.output.with_suffix(".html")
    json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(make_html(result.title, payload, final), encoding="utf-8")
    print("VERSÃO:", ExplanationGenerator.VERSION)
    print("MODELO CONFIGURADO:", os.getenv("HIBRIA_QWEN_MODEL") or ExplanationGenerator.DEFAULT_MODEL)
    print("ENTRADA DO MODELO:", trace.get("input_diagnostics"))
    if args.show_prompt:
        print(json.dumps((trace.get("attempts") or [{}])[0].get("payload") or trace.get("request_preview"), ensure_ascii=False, indent=2))
    for i, attempt in enumerate(trace.get("attempts", []), 1):
        print(f"\nRESPOSTA BRUTA — tentativa {i}:")
        print(attempt.get("raw", "Sem resposta"))
        print("VALIDAÇÃO:", json.dumps(attempt.get("validation"), ensure_ascii=False))
        print("MÉTRICAS:", json.dumps(attempt.get("metrics"), ensure_ascii=False))
    print("\nRESULTADO FINAL:", json.dumps(final, ensure_ascii=False, indent=2))
    print("\nPRÉVIA HTML:", html_path.resolve())
    print("DIAGNÓSTICO JSON:", json_path.resolve())
    if args.save_explanation:
        save_explanation(args.analysis_id, payload, final)
        print("Somente a explicação foi atualizada. Nota, rótulo, ID e avaliações foram preservados.")
    else:
        print("Banco não alterado.")
    print("Nenhuma API de pesquisa foi consultada.")


if __name__ == "__main__":
    main()
