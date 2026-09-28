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

from pipeline.output.explanation_generator import ExplanationGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Testa a explicacao v3 com uma analise salva, sem pesquisar "
            "novamente e sem alterar o banco."
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
    saved_stances = [
        item
        for item in (transparency.get("stance_results") or [])
        if isinstance(item, dict)
    ]

    stances_by_claim: dict[str, list[dict[str, Any]]] = {}
    for item in saved_stances:
        claim_id = str(item.get("claim_id") or "")
        stances_by_claim.setdefault(claim_id, []).append(item)

    retrieval_results: list[SimpleNamespace] = []
    for claim_data in evidence_block.get("claims") or []:
        claim_id = str(claim_data.get("claim_id") or "")
        claim = SimpleNamespace(
            claim_id=claim_id,
            text=str(claim_data.get("text") or ""),
        )
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
                    stance=(
                        evidence_data.get("stance")
                        or saved_stance.get("stance")
                    ),
                )
            )

        retrieval_results.append(
            SimpleNamespace(
                claim=claim,
                evidences=evidences,
            )
        )

    return SimpleNamespace(
        title=str(analysis.get("title") or ""),
        label_final=analysis.get("label"),
        score_final=analysis.get("score"),
        score_breakdown=analysis.get("score_breakdown") or {},
        retrieval_results=retrieval_results,
        stance_results=saved_stances,
    )


def request_qwen(result: SimpleNamespace) -> tuple[str, dict[str, Any]]:
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
    context = ExplanationGenerator._build_context(result)
    prompt = ExplanationGenerator._build_prompt(result, context=context)
    payload = {
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
            "temperature": 0.2,
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
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    answer = response.json()
    raw_content = str(
        (answer.get("message") or {}).get("content") or ""
    ).strip()
    return raw_content, context


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=True)

    payload = load_analysis(args.analysis_id)
    result = build_result(payload)
    raw_content, context = request_qwen(result)
    fallback = ExplanationGenerator._fallback_report_from_context(
        result,
        context,
    )

    print("MODELO:", os.getenv("HIBRIA_QWEN_MODEL") or ExplanationGenerator.DEFAULT_MODEL)
    print("\nFICHA CONTROLADA ENVIADA AO QWEN:")
    print(json.dumps(context, ensure_ascii=False, indent=2))
    print("\nRESPOSTA DO QWEN ANTES DA VALIDACAO:")
    print(raw_content)

    parsed = ExplanationGenerator._parse_report(raw_content)
    if parsed is None:
        final = {**fallback, "source": "deterministic"}
        explanation_accepted = False
        details_accepted = False
    else:
        final = ExplanationGenerator._validate_model_report(
            parsed,
            result=result,
            context=context,
            fallback=fallback,
        )
        explanation_accepted = final["explanation"] == parsed["explanation"]
        details_accepted = final["details"] == parsed["details"]

    print("\nVALIDACAO DA HIBRIA:")
    print(
        "Explicacao do Qwen:",
        "ACEITA" if explanation_accepted else "SUBSTITUIDA",
    )
    print(
        "Detalhes do Qwen:",
        "ACEITOS" if details_accepted else "SUBSTITUIDOS",
    )
    print("\nRESULTADO FINAL:")
    print("ORIGEM:", final["source"])
    print("\nEVIDENCIAS:")
    print(final["explanation"])
    print("\nDETALHES:")
    for detail in final["details"]:
        print(f"- {detail}")
    print("\nJSON FINAL:")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print("\nNenhuma API de pesquisa foi consultada e o banco nao foi alterado.")


if __name__ == "__main__":
    main()
