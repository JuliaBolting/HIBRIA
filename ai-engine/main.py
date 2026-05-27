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

    # Classificação da qualidade da extração:
    #
    # success → conteúdo suficiente para seguir no pipeline
    # partial → pouco conteúdo; análise pode ficar incompleta
    # empty   → nenhum bloco textual extraído; só metadados
    
    extraction_status = "success"

    if result.block_count == 0 or result.char_count == 0:
        extraction_status = "empty"
        result.warnings.append(
            "Nenhum bloco textual foi extraído; apenas metadados da página foram encontrados."
        )

    elif result.char_count < 1000:
        extraction_status = "partial"
        result.warnings.append(
            "Pouco conteúdo textual foi extraído; a análise pode ficar incompleta."
        )

    # Guarda o status no objeto, mesmo que o campo ainda não exista formalmente
    # no PipelineResult. O to_dict() precisa incluir esse campo para ele aparecer no JSON.
    result.extraction_status = extraction_status

    print(
        f"[extractor]  {result.block_count} blocos · "
        f"{result.char_count} chars · via {result.render_method} · status={extraction_status}"
    )

    print(
        f"[cleaner]    {len(result.blocks_clean)} blocos limpos"
    )

    print(
        f"[normalizer] {len(result.blocks_bert)} blocos BERT · "
        f"{len(result.blocks_tfidf)} blocos TF-IDF · "
        f"{len(result.blocks_similarity)} blocos similarity"
    )

    print(
        f"[segmenter]  {result.sentence_count} sentenças · "
        f"{result.segment_count} segmentos"
    )

    if result.segmentation_stats:
        print(
            f"[segmenter]  {result.segmentation_stats.get('total_token_count', 0)} tokens · "
            f"média {result.segmentation_stats.get('avg_tokens_per_sent', 0)} tokens/sentença"
        )

    if result.warnings:
        for warning in result.warnings:
            print(f"[aviso] {warning}")

    output = result.to_dict()

    # Garante que o status apareça no JSON mesmo antes de ajustar o PipelineResult
    output["extraction_status"] = extraction_status

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSalvo em {OUTPUT_PATH}")

except ExtractionError as e:
    print(f"[ERRO] {e}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"[ERRO inesperado] {e}", file=sys.stderr)
    sys.exit(1)