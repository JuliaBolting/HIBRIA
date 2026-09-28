from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import psycopg2
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.analysis.stance_model import StanceModel
from pipeline.output.explanation_generator import ExplanationGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o fluxo real de revisao sobre uma analise salva, "
            "mostra a resposta bruta do Qwen e a versao apos validacao."
        )
    )
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument(
        "--env-file",
        default="/home/ubuntu/.config/hibria/hibria.env",
    )
    return parser.parse_args()


def load_analysis(analysis_id: str) -> dict[str, Any]:
    database_url = os.getenv("HIBRIA_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("HIBRIA_DATABASE_URL nao esta configurada.")

    with psycopg2.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT resultado_json FROM analises WHERE id = %s;",
                (analysis_id,),
            )
            row = cursor.fetchone()

    if not row:
        raise RuntimeError(f"Analise nao encontrada: {analysis_id}")

    payload = row[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("resultado_json possui formato invalido.")
    return payload


def build_result(payload: dict[str, Any]) -> SimpleNamespace:
    analysis = payload.get("analysis") or {}
    evidence_block = payload.get("evidence") or {}
    transparency = payload.get("transparency") or {}
    saved_stances = list(transparency.get("stance_results") or [])

    stances_by_claim: dict[str, list[dict[str, Any]]] = {}
    for item in saved_stances:
        if isinstance(item, dict):
            claim_id = str(item.get("claim_id") or "")
            stances_by_claim.setdefault(claim_id, []).append(item)

    claims: list[SimpleNamespace] = []
    retrieval_results: list[SimpleNamespace] = []

    for claim_data in evidence_block.get("claims") or []:
        claim_id = str(claim_data.get("claim_id") or "")
        claim = SimpleNamespace(
            claim_id=claim_id,
            text=str(claim_data.get("text") or ""),
            normalized=str(claim_data.get("text") or ""),
        )
        claims.append(claim)
        evidences: list[SimpleNamespace] = []

        for evidence_data in claim_data.get("evidences") or []:
            source = str(evidence_data.get("source") or "")
            url = str(evidence_data.get("url") or "")
            saved_stance = next(
                (
                    item
                    for item in stances_by_claim.get(claim_id, [])
                    if (
                        (url and str(item.get("url") or "") == url)
                        or (source and str(item.get("source") or "") == source)
                    )
                ),
                {},
            )
            evidences.append(
                SimpleNamespace(
                    evidence_id=str(
                        evidence_data.get("evidence_id")
                        or saved_stance.get("evidence_id")
                        or ""
                    ),
                    text=str(evidence_data.get("text") or ""),
                    source=source,
                    title=str(evidence_data.get("title") or ""),
                    url=url,
                    domain=str(evidence_data.get("domain") or ""),
                    similarity=float(evidence_data.get("similarity") or 0.0),
                    published_at=evidence_data.get("published_at"),
                    retrieval_layer=str(evidence_data.get("retrieval_layer") or ""),
                    source_type=str(evidence_data.get("source_type") or ""),
                    trusted_source=bool(evidence_data.get("trusted_source")),
                    stance=evidence_data.get("stance"),
                    metadata=evidence_data.get("metadata") or {},
                )
            )

        retrieval_results.append(
            SimpleNamespace(
                claim=claim,
                evidences=evidences,
                layers_used=[],
                layers_failed={},
                retrieval_time=0.0,
            )
        )

    # Mantem o mesmo preparo usado nos testes comparativos anteriores.
    stance_results = StanceModel.analyze_retrieval_results(retrieval_results)

    return SimpleNamespace(
        title=str(analysis.get("title") or ""),
        label_final=analysis.get("label"),
        score_final=analysis.get("score"),
        score_breakdown=analysis.get("score_breakdown") or {},
        reputation=(payload.get("source") or {}).get("reputation") or {},
        claims=claims,
        retrieval_results=retrieval_results,
        stance_results=stance_results,
    )


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=True)

    payload = load_analysis(args.analysis_id)
    result = build_result(payload)
    base_report = ExplanationGenerator.fallback_report(result)

    # Replica exatamente a chamada real, mas imprime a resposta antes de
    # aplicar a validação e a contingência determinística.
    api_url = os.getenv(
        "HIBRIA_QWEN_API_URL",
        ExplanationGenerator.DEFAULT_API_URL,
    ).strip() or ExplanationGenerator.DEFAULT_API_URL
    model = os.getenv(
        "HIBRIA_QWEN_MODEL",
        ExplanationGenerator.DEFAULT_MODEL,
    ).strip() or ExplanationGenerator.DEFAULT_MODEL
    timeout = ExplanationGenerator._env_int(
        "HIBRIA_QWEN_TIMEOUT_SECONDS",
        ExplanationGenerator.DEFAULT_TIMEOUT,
    )
    prompt = ExplanationGenerator._build_prompt(
        result,
        base_report=base_report,
    )
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": ExplanationGenerator._system_prompt(),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "think": False,
        "format": ExplanationGenerator.OUTPUT_SCHEMA,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "num_ctx": 4096,
            "num_predict": ExplanationGenerator._env_int(
                "HIBRIA_QWEN_MAX_OUTPUT_TOKENS",
                420,
            ),
        },
        "keep_alive": "10m",
    }
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        json=request_payload,
        timeout=timeout,
    )
    response.raise_for_status()
    ollama_payload = response.json()
    raw_content = str(
        (ollama_payload.get("message") or {}).get("content") or ""
    ).strip()

    print("MODELO:", model)
    print("ORIGEM: qwen bruto (sem validacao da HIBRIA)")
    print("\nRESPOSTA BRUTA:")
    print(raw_content)

    report = ExplanationGenerator._parse_report(raw_content)
    if report is None:
        explanation_accepted = False
        details_accepted = False
        report = {
            **base_report,
            "source": "deterministic",
        }
    else:
        explanation_accepted = ExplanationGenerator._is_safe_rewrite(
            report["explanation"],
            reference=base_report["explanation"],
            label=str(result.label_final or ""),
            require_label=True,
        )
        details_accepted = ExplanationGenerator._are_safe_rewritten_details(
            report["details"],
            reference=base_report["details"],
            explanation=(
                report["explanation"]
                if explanation_accepted
                else base_report["explanation"]
            ),
            label=str(result.label_final or ""),
        )

        if not explanation_accepted:
            report["explanation"] = base_report["explanation"]
        if not details_accepted:
            report["details"] = base_report["details"]

        if explanation_accepted and details_accepted:
            report["source"] = "qwen"
        elif explanation_accepted or details_accepted:
            report["source"] = "hybrid"
        else:
            report["source"] = "deterministic"

    print("\nVALIDACAO DA HIBRIA:")
    print(
        "Explicacao do Qwen:",
        "ACEITA" if explanation_accepted else "SUBSTITUIDA",
    )
    print(
        "Detalhes do Qwen:",
        "ACEITOS" if details_accepted else "SUBSTITUIDOS",
    )

    print("\nDEPOIS DA HIBRIA:")
    print("ORIGEM:", report["source"])
    print("\nEVIDENCIAS:")
    print(report.get("explanation") or "")
    print("\nDETALHES:")
    for detail in report.get("details") or []:
        print(f"- {detail}")

    print("\nJSON FINAL:")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\nNenhuma API de pesquisa foi consultada e o banco nao foi alterado.")


if __name__ == "__main__":
    main()
