# =============================================================================
# bertimbau_classifier.py
# Classificação textual com BERTimbau fine-tuned para notícias falsas.
#
# Este módulo usa um modelo SequenceClassification do Hugging Face. Por padrão,
# foi configurado para um checkpoint BERTimbau treinado no Fake.Br Corpus, mas
# o modelo pode ser trocado por variável de ambiente.
#
# Saída principal:
#   result.classification = {
#       "status": "ok",
#       "label": "confiável" | "não confiável" | "indefinido",
#       "score": 0.0-1.0,              # probabilidade de notícia verdadeira
#       "fake_probability": 0.0-1.0,
#       "real_probability": 0.0-1.0,
#       "confidence": 0.0-1.0,
#       ...
#   }
# =============================================================================

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "vzani/portuguese-fake-news-classifier-bertimbau-fake-br"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "sim",
        "s",
        "on",
    }


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning(
            "[bertimbau] valor inválido para %s=%r; usando padrão %s",
            name,
            value,
            default,
        )
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        logger.warning(
            "[bertimbau] valor inválido para %s=%r; usando padrão %s",
            name,
            value,
            default,
        )
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_label(label: str) -> str:
    """Normaliza nomes de labels vindos de modelos diferentes."""
    label = (label or "").strip().lower()
    label = label.replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^a-z0-9_áéíóúàâêôãõç]", "", label)
    return label


def split_env_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name, "")

    if not raw.strip():
        return {normalize_label(item) for item in default}

    return {normalize_label(item) for item in raw.split(",") if normalize_label(item)}


class BERTimbauClassifier:
    """
    Classificador de notícias usando BERTimbau fine-tuned.

    Observação importante:
    BERTimbau base é apenas um modelo pré-treinado. Para classificar fake/true,
    é necessário usar um checkpoint fine-tuned para classificação binária.
    """

    _model: Any | None = None
    _tokenizer: Any | None = None
    _torch: Any | None = None
    _loaded_model_name: str | None = None
    _loaded_device: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.enabled = env_flag("HIBRIA_ENABLE_BERTIMBAU", True)
        self.model_name = (
            model_name
            or os.getenv("HIBRIA_BERTIMBAU_MODEL", "").strip()
            or DEFAULT_MODEL_NAME
        )

        # O BERT usa até 512 tokens; o limite em caracteres evita textos enormes
        # antes da tokenização e mantém a análise responsiva na API.
        self.max_chars_per_chunk = env_int("HIBRIA_BERTIMBAU_MAX_CHARS_PER_CHUNK", 1800)
        self.max_chunks = env_int("HIBRIA_BERTIMBAU_MAX_CHUNKS", 8)
        self.max_tokens = env_int("HIBRIA_BERTIMBAU_MAX_TOKENS", 512)
        self.device_name = os.getenv("HIBRIA_BERTIMBAU_DEVICE", "auto").strip().lower()
        self.local_files_only = env_flag("HIBRIA_BERTIMBAU_LOCAL_FILES_ONLY", False)
        self.trust_remote_code = env_flag("HIBRIA_BERTIMBAU_TRUST_REMOTE_CODE", False)

        self.true_threshold = env_float("HIBRIA_BERTIMBAU_TRUE_THRESHOLD", 0.60)
        self.fake_threshold = env_float("HIBRIA_BERTIMBAU_FAKE_THRESHOLD", 0.60)

        # No checkpoint usado como padrão, LABEL_1 = true e LABEL_0 = fake.
        # As variáveis abaixo permitem corrigir o mapeamento sem alterar código.
        self.fake_labels = split_env_set(
            "HIBRIA_BERTIMBAU_FAKE_LABELS",
            {"LABEL_0", "fake", "falso", "falsa", "nao_confiavel", "não_confiável"},
        )
        self.real_labels = split_env_set(
            "HIBRIA_BERTIMBAU_REAL_LABELS",
            {
                "LABEL_1",
                "true",
                "real",
                "verdadeiro",
                "verdadeira",
                "confiavel",
                "confiável",
            },
        )

    # ---------------------------------------------------------------------
    # Preparação do texto
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text

    def _iter_clean_blocks(self, blocks: Iterable[str] | str | None) -> list[str]:
        if blocks is None:
            return []

        if isinstance(blocks, str):
            blocks = [blocks]

        cleaned: list[str] = []

        for block in blocks:
            text = self._normalize_text(str(block))
            if text:
                cleaned.append(text)

        return cleaned

    def _split_long_block(self, text: str) -> list[str]:
        if len(text) <= self.max_chars_per_chunk:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.max_chars_per_chunk:
                if current:
                    chunks.append(current.strip())
                    current = ""

                for i in range(0, len(sentence), self.max_chars_per_chunk):
                    chunks.append(sentence[i : i + self.max_chars_per_chunk].strip())
                continue

            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= self.max_chars_per_chunk:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

        return chunks

    def _build_chunks(self, blocks: Iterable[str] | str | None) -> list[str]:
        cleaned_blocks = self._iter_clean_blocks(blocks)

        if not cleaned_blocks:
            return []

        chunks: list[str] = []
        current = ""

        for block in cleaned_blocks:
            pieces = self._split_long_block(block)

            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip()

                if len(candidate) <= self.max_chars_per_chunk:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = piece

                if len(chunks) >= self.max_chunks:
                    break

            if len(chunks) >= self.max_chunks:
                break

        if current and len(chunks) < self.max_chunks:
            chunks.append(current)

        return chunks[: self.max_chunks]

    # ---------------------------------------------------------------------
    # Modelo
    # ---------------------------------------------------------------------

    def _resolve_device(self, torch_module: Any) -> Any:
        if self.device_name == "cpu":
            return torch_module.device("cpu")

        if self.device_name == "cuda":
            if torch_module.cuda.is_available():
                return torch_module.device("cuda")
            logger.warning("[bertimbau] CUDA solicitada, mas indisponível; usando CPU")
            return torch_module.device("cpu")

        # auto
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        return torch_module.device("cpu")

    def _ensure_loaded(self) -> tuple[Any, Any, Any, Any]:
        if (
            self.__class__._model is not None
            and self.__class__._tokenizer is not None
            and self.__class__._torch is not None
            and self.__class__._loaded_model_name == self.model_name
        ):
            device = self._resolve_device(self.__class__._torch)
            return (
                self.__class__._torch,
                self.__class__._tokenizer,
                self.__class__._model,
                device,
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depende do ambiente local
            raise RuntimeError(
                "Dependências ausentes. Instale torch e transformers para usar o BERTimbau."
            ) from exc

        device = self._resolve_device(torch)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        model.to(device)
        model.eval()

        self.__class__._torch = torch
        self.__class__._tokenizer = tokenizer
        self.__class__._model = model
        self.__class__._loaded_model_name = self.model_name
        self.__class__._loaded_device = str(device)

        return torch, tokenizer, model, device

    # ---------------------------------------------------------------------
    # Classificação
    # ---------------------------------------------------------------------

    def _predict_chunk(self, text: str) -> list[dict[str, float | str]]:
        torch, tokenizer, model, device = self._ensure_loaded()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = (
                torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()
            )

        id2label = getattr(model.config, "id2label", {}) or {}

        return [
            {
                "label": str(id2label.get(index, f"LABEL_{index}")),
                "score": round(float(probability), 6),
            }
            for index, probability in enumerate(probabilities)
        ]

    def _summarize_scores(
        self,
        chunk_predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        real_weighted = 0.0
        fake_weighted = 0.0
        total_weight = 0.0
        unknown_labels: set[str] = set()

        for prediction in chunk_predictions:
            weight = max(1, int(prediction.get("weight", 1)))
            total_weight += weight

            for item in prediction.get("scores", []):
                label = normalize_label(str(item.get("label", "")))
                score = float(item.get("score", 0.0))

                if label in self.real_labels:
                    real_weighted += score * weight
                elif label in self.fake_labels:
                    fake_weighted += score * weight
                else:
                    unknown_labels.add(label)

        if total_weight <= 0:
            return {
                "status": "error",
                "message": "Nenhuma predição válida foi gerada pelo BERTimbau.",
            }

        real_probability = real_weighted / total_weight
        fake_probability = fake_weighted / total_weight

        # Alguns modelos retornam só a classe predita. Quando uma das classes não
        # aparece, a probabilidade complementar evita quebrar a agregação.
        if real_probability == 0 and fake_probability > 0:
            real_probability = 1.0 - fake_probability
        elif fake_probability == 0 and real_probability > 0:
            fake_probability = 1.0 - real_probability

        real_probability = clamp(real_probability)
        fake_probability = clamp(fake_probability)
        confidence = max(real_probability, fake_probability)

        if (
            real_probability >= self.true_threshold
            and real_probability >= fake_probability
        ):
            label = "confiável"
        elif (
            fake_probability >= self.fake_threshold
            and fake_probability > real_probability
        ):
            label = "não confiável"
        else:
            label = "indefinido"

        return {
            "status": "ok",
            "label": label,
            "score": round(real_probability, 4),
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_probability, 4),
            "real_probability": round(real_probability, 4),
            "unknown_labels": sorted(unknown_labels),
        }

    def classify(self, blocks: Iterable[str] | str | None) -> dict[str, Any]:
        """
        Classifica os blocos normalizados para BERT.

        O score retornado representa a probabilidade de o texto ser classificado
        como notícia verdadeira pelo modelo fine-tuned. Assim, o aggregator pode
        combinar esse componente com RAG, stance e reputação.
        """
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "model_name": self.model_name,
                "message": "BERTimbau desativado por HIBRIA_ENABLE_BERTIMBAU=false.",
            }

        chunks = self._build_chunks(blocks)

        if not chunks:
            return {
                "enabled": True,
                "status": "no_text",
                "model_name": self.model_name,
                "message": "Nenhum texto disponível para classificação com BERTimbau.",
            }

        try:
            chunk_predictions: list[dict[str, Any]] = []

            for index, chunk in enumerate(chunks, start=1):
                scores = self._predict_chunk(chunk)
                chunk_predictions.append(
                    {
                        "chunk": index,
                        "chars": len(chunk),
                        "weight": len(chunk),
                        "scores": scores,
                    }
                )

            summary = self._summarize_scores(chunk_predictions)
            summary.update(
                {
                    "enabled": True,
                    "model_name": self.model_name,
                    "device": self.__class__._loaded_device,
                    "chunks_analyzed": len(chunk_predictions),
                    "max_chunks": self.max_chunks,
                    "max_tokens": self.max_tokens,
                    "probabilities": {
                        "confiavel": summary.get("real_probability"),
                        "nao_confiavel": summary.get("fake_probability"),
                    },
                    "chunk_predictions": chunk_predictions,
                }
            )
            return summary

        except Exception as exc:
            logger.error("[bertimbau] falha na classificação", exc_info=True)
            return {
                "enabled": True,
                "status": "error",
                "model_name": self.model_name,
                "message": f"{type(exc).__name__}: {exc}",
            }
