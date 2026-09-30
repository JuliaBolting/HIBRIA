"""Entrada de linha de comando. A API de produção permanece api.app:app."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Analisar uma URL com a HÍBRIA")
    parser.add_argument("url", help="URL pública da notícia")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/analysis.json"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    from pipeline.pipeline import HibriaPipeline
    result = HibriaPipeline.run(args.url)
    payload = result.response or result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultado: {result.label_final} — índice {result.score_final}")
    print(f"Arquivo: {args.output.resolve()}")


if __name__ == "__main__":
    main()
