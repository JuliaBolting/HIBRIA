from dataclasses import dataclass, field
from typing import Literal

from pipeline.preprocessing import (
    TextExtractor,
    TextCleaner,
    TextNormalizer,
    ExtractionError,
)


@dataclass
class PipelineResult:
    """
    Contrato de dados entre todas as etapas do pipeline.
    Cada módulo lê o que precisa e escreve no seu campo.
    Campos ainda não implementados ficam como None — o pipeline
    continua funcionando parcialmente enquanto o projeto cresce.
    """
    # ── extractor ──────────────────────────────────────────
    url:               str = ""
    title:             str = ""
    description:       str = ""
    render_method:     str = ""        # "static" | "playwright"
    paywall_detected:  bool = False
    warnings:          list[str] = field(default_factory=list)

    # ── cleaner ────────────────────────────────────────────
    blocks_clean:      list[str] = field(default_factory=list)

    # ── normalizer ─────────────────────────────────────────
    blocks_bert:       list[str] = field(default_factory=list)
    blocks_tfidf:      list[str] = field(default_factory=list)
    blocks_similarity: list[str] = field(default_factory=list)

    # ── segmentation (a implementar) ───────────────────────
    segments:          list[str] | None = None

    # ── claim_detector (a implementar) ─────────────────────
    claims:            list[str] | None = None

    # ── verificação híbrida (a implementar) ────────────────
    score:             float | None = None
    label:             str | None = None       # "verdadeiro" | "falso" | "inconclusivo"
    evidence:          list[dict] | None = None

    # ── explanation_generator (a implementar) ──────────────
    explanation:       str | None = None

    # ── métricas ───────────────────────────────────────────
    @property
    def block_count(self) -> int:
        return len(self.blocks_clean)

    @property
    def char_count(self) -> int:
        return sum(len(b) for b in self.blocks_clean)

    def to_dict(self) -> dict:
        return {
            "url":               self.url,
            "title":             self.title,
            "description":       self.description,
            "render_method":     self.render_method,
            "paywall_detected":  self.paywall_detected,
            "warnings":          self.warnings,
            "block_count":       self.block_count,
            "char_count":        self.char_count,
            "blocks_clean":      self.blocks_clean,
            "blocks_bert":       self.blocks_bert,
            "blocks_tfidf":      self.blocks_tfidf,
            "blocks_similarity": self.blocks_similarity,
            "segments":          self.segments,
            "claims":            self.claims,
            "score":             self.score,
            "label":             self.label,
            "evidence":          self.evidence,
            "explanation":       self.explanation,
        }


class PreprocessingPipeline:
    """
    Orquestra extractor → cleaner → normalizer.
    Cada etapa recebe e devolve um PipelineResult,
    tornando simples adicionar ou reordenar etapas.
    """

    @staticmethod
    def _step_extract(url: str, result: PipelineResult) -> PipelineResult:
        raw = TextExtractor.extract(url)

        result.url              = raw["url"]
        result.title            = raw["title"]
        result.description      = raw["description"]
        result.render_method    = raw["render_method"]
        result.paywall_detected = raw["paywall_detected"]
        result.warnings         = raw["warnings"]

        # guarda os blocos brutos temporariamente para o próximo passo
        result._raw_blocks = raw["content_blocks"]  # removido após limpeza
        return result

    @staticmethod
    def _step_clean(result: PipelineResult) -> PipelineResult:
        raw_blocks = getattr(result, "_raw_blocks", [])

        result.blocks_clean = TextCleaner.clean_blocks(
            raw_blocks,
            source=result.render_method,
        )

        del result._raw_blocks  # libera memória — não é mais necessário
        return result

    @staticmethod
    def _step_normalize(result: PipelineResult) -> PipelineResult:
        result.blocks_bert = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="bert"
        )
        result.blocks_tfidf = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="tfidf"
        )
        result.blocks_similarity = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="similarity"
        )
        return result

    @classmethod
    def run(cls, url: str) -> PipelineResult:
        result = PipelineResult()

        steps = [
            ("extractor",   lambda r: cls._step_extract(url, r)),
            ("cleaner",     cls._step_clean),
            ("normalizer",  cls._step_normalize),
            # ("segmentation",  cls._step_segment),   ← adiciona quando implementar
            # ("claim_detector", cls._step_detect),   ← idem
        ]

        for name, step in steps:
            try:
                result = step(result)
            except ExtractionError:
                raise  # erros de extração sobem direto — são fatais
            except Exception as e:
                # erros nas etapas de processamento viram warnings,
                # o pipeline continua com o que tem até aquele ponto
                result.warnings.append(f"[{name}] falhou: {e}")

        return result