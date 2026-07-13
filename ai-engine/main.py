import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

print(f"Inicializando HÍBRIA...")

from pipeline.preprocessing.normalization import TextNormalizer
from pipeline.analysis.embeddings import EmbeddingModel

TextNormalizer._get_nlp()       # carrega spaCy na RAM
EmbeddingModel().warmup()       # carrega SentenceTransformer na RAM

print("Modelos prontos.\n")

from pipeline.pipeline import HibriaPipeline
from pipeline.preprocessing.extractor import ExtractionError

OUTPUT_PATH = Path(__file__).parent.parent / "extension" / "public" / "output.json"
URL = "https://ndmais.com.br/economia/negocios/sorveteria-de-60-anos-fecha-para-mudanca-historica-em-blumenau/"

try:
    result = HibriaPipeline.run(URL)

    # log resumido no terminal
    print(f"\n{'='*60}")
    print(f"  HÍBRIA — Resultado da Análise")
    print(f"{'='*60}")
    print(f"  URL:        {result.url[:70]}...")
    print(f"  Título:     {result.title[:70]}")
    print(f"  Método:     {result.render_method}")
    print(f"  Blocos:     {result.block_count} ({result.char_count} chars)")
    print(f"  Sentenças:  {len(result.sentences or [])}")
    print(f"  Claims:     {result.claim_count}")
    print(f"  Evidências: {result.evidence_count}")
    print(f"  Score:      {result.score_final or 'pendente'}")
    print(f"  Label:      {result.label_final or 'pendente'}")
    print(f"{'='*60}")
    print(f"  Tempo por etapa:")
    for step, t in result._processing_time.items():
        print(f"    {step:<20} {t:.3f}s")
    print(f"{'='*60}\n")

    if result.warnings:
        print("Avisos:")
        for w in result.warnings:
            print(f"  ⚠ {w}")
        print()

    # salva JSON completo
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result.to_dict(), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Salvo em {OUTPUT_PATH}")

except ExtractionError as e:
    print(f"[ERRO] {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"[ERRO inesperado] {e}", file=sys.stderr)
    raise