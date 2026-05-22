import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pipeline.pipeline import PreprocessingPipeline
from pipeline.preprocessing import ExtractionError

OUTPUT_PATH = Path("extension/public/output.json")
URL = "https://g1.globo.com/sp/bauru-marilia/noticia/2026/05/11/policia-encontra-cerveja-e-cooler-em-carro-de-motorista-suspeito-de-provocar-acidente-que-matou-quatro-jovens-no-dia-das-maes.ghtml"

try:
    result = PreprocessingPipeline.run(URL)

    print(f"[extractor]  {result.block_count} blocos · {result.char_count} chars · via {result.render_method}")
    print(f"[cleaner]    {len(result.blocks_clean)} blocos limpos")
    print(f"[normalizer] {len(result.blocks_bert)} blocos normalizados")

    if result.warnings:
        for w in result.warnings:
            print(f"[aviso] {w}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result.to_dict(), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSalvo em {OUTPUT_PATH}")

except ExtractionError as e:
    print(f"[ERRO] {e}", file=sys.stderr)
    sys.exit(1)