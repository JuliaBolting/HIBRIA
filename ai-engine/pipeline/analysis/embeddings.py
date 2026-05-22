# =============================================================================
# embeddings.py
# Geração de representações vetoriais (embeddings) para texto.
#
# É o módulo central que conecta:
#   retrieval/   → vector_store.py usa embeddings para indexar documentos
#                → retriever.py usa embeddings para busca semântica
#   analysis/    → similarity.py usa embeddings para calcular distância
#                → stance_model.py usa embeddings para comparar claim x evidência
#
# Modelo padrão: paraphrase-multilingual-MiniLM-L12-v2
#   - Multilíngue (50+ idiomas, incluindo português)
#   - Leve: 118MB, 384 dimensões
#   - Rápido: ~14k sentenças/segundo em CPU
#   - Bom equilíbrio qualidade/velocidade para TCC
#
# Upgrade futuro sem refatoração:
#   - BERTimbau-large para embeddings em português puro
#   - text-embedding-3-small (OpenAI) via API
#   - e5-multilingual-large para maior precisão
#
# Entrada:  str | list[str]
# Saída:    np.ndarray  shape=(384,) | shape=(N, 384)
# =============================================================================

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuração de modelos disponíveis
#
# Estruturado como dicionário para facilitar troca de modelo sem
# alterar o código dos módulos que consomem embeddings.
# Basta mudar ACTIVE_MODEL em .env ou na inicialização.
# =============================================================================

AVAILABLE_MODELS = {
    # padrão: multilíngue leve — sem necessidade de fine-tuning em PT
    "multilingual-minilm": {
        "model_id":  "paraphrase-multilingual-MiniLM-L12-v2",
        "dims":      384,
        "max_tokens": 128,
        "language":  "multilingual",
        "size_mb":   118,
    },
    # upgrade: multilíngue mais preciso — 2x mais lento, melhor qualidade
    "multilingual-mpnet": {
        "model_id":  "paraphrase-multilingual-mpnet-base-v2",
        "dims":      768,
        "max_tokens": 128,
        "language":  "multilingual",
        "size_mb":   278,
    },
    # upgrade pt-BR: BERTimbau fine-tuned para similaridade semântica
    "bertimbau-sts": {
        "model_id":  "rufimelo/bert-large-portuguese-cased-sts",
        "dims":      1024,
        "max_tokens": 512,
        "language":  "pt",
        "size_mb":   1340,
    },
}

DEFAULT_MODEL = "multilingual-minilm"


# =============================================================================
# Cache de embeddings em memória
#
# Evita recomputar embeddings para textos idênticos dentro da mesma sessão.
# Crítico para o pipeline: claims e segmentos são embedados múltiplas vezes
# (pelo retriever, pelo similarity e pelo stance_model).
#
# Implementação simples com dict — sem LRU por enquanto.
# FUTURO: persistir cache em disco (joblib ou shelve) para reutilizar
# entre sessões, especialmente para a base vetorial local.
# =============================================================================

class EmbeddingCache:
    """
    Cache em memória de embeddings por hash do texto.
    Thread-safe para uso futuro com processamento paralelo.
    """

    def __init__(self, max_size: int = 5000):
        self._cache: dict[str, np.ndarray] = {}
        self._max_size = max_size
        self._hits   = 0
        self._misses = 0

    @staticmethod
    def _key(text: str, model_id: str) -> str:
        """Hash SHA-256 do texto + modelo — chave única por conteúdo + modelo."""
        raw = f"{model_id}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str, model_id: str) -> np.ndarray | None:
        key = self._key(text, model_id)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def set(self, text: str, model_id: str, embedding: np.ndarray) -> None:
        # descarta entradas antigas se atingiu o limite (FIFO simples)
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._key(text, model_id)
        self._cache[key] = embedding

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size":      len(self._cache),
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / total, 3) if total else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits   = 0
        self._misses = 0


# =============================================================================
# EmbeddingModel — wrapper sobre SentenceTransformers
# =============================================================================

class EmbeddingModel:
    """
    Wrapper sobre SentenceTransformers com cache, batching e normalização.

    Singleton por model_name — garante que o modelo pesado seja carregado
    uma única vez por processo, mesmo que múltiplos módulos instanciem
    EmbeddingModel independentemente.
    """

    # registro de instâncias por model_name (singleton por modelo)
    _instances: dict[str, "EmbeddingModel"] = {}

    def __new__(cls, model_name: str = DEFAULT_MODEL) -> "EmbeddingModel":
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # __init__ é chamado toda vez que EmbeddingModel() é instanciado
        # mas __new__ garante que é a mesma instância — evita reinicialização
        if self._initialized:
            return

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. "
                f"Disponíveis: {list(AVAILABLE_MODELS.keys())}"
            )

        self._model_name   = model_name
        self._model_config = AVAILABLE_MODELS[model_name]
        self._model        = None   # lazy loading — carrega na primeira chamada
        self._cache        = EmbeddingCache()
        self._initialized  = True

        logger.info(
            f"[embeddings] modelo configurado: {self._model_config['model_id']} "
            f"({self._model_config['dims']}d, {self._model_config['size_mb']}MB)"
        )

    # =========================================================================
    # Lazy loading do modelo
    # =========================================================================

    def _get_model(self):
        """
        Carrega o modelo SentenceTransformer na primeira chamada.

        Por que lazy loading:
          - O modelo leva 2-5s para carregar e ~120MB de RAM
          - Módulos que importam embeddings.py não devem pagar esse custo
            se não chegarem a gerar embeddings (ex: pipeline com erro na extração)
          - Permite importar o módulo sem ter sentence-transformers instalado
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers não instalado. Execute:\n"
                    "  pip install sentence-transformers"
                )

            model_id = self._model_config["model_id"]
            logger.info(f"[embeddings] carregando modelo: {model_id}")
            t0 = time.time()

            self._model = SentenceTransformer(model_id)

            logger.info(
                f"[embeddings] modelo carregado em {time.time() - t0:.2f}s"
            )

        return self._model

    # =========================================================================
    # Geração de embeddings
    # =========================================================================

    def embed(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Gera embedding para um único texto.

        normalize=True: aplica L2 normalization ao vetor resultante.
          Obrigatório para que similaridade cosine seja computável como
          produto escalar simples (dot product) — muito mais rápido no FAISS.
          Desativar apenas se o módulo downstream precisar do vetor bruto.

        Retorno: np.ndarray shape=(dims,) dtype=float32
        """
        text = text.strip()
        if not text:
            # vetor zero para texto vazio — não quebra o pipeline downstream
            return np.zeros(self._model_config["dims"], dtype=np.float32)

        # verifica cache antes de computar
        model_id = self._model_config["model_id"]
        cached = self._cache.get(text, model_id)
        if cached is not None:
            return cached

        model  = self._get_model()
        vector = model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)

        self._cache.set(text, model_id, vector)
        return vector

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos com batching eficiente.

        Estratégia de cache em batch:
          1. Verifica cache para cada texto individualmente
          2. Agrupa os textos sem cache em um único batch para o modelo
          3. Preenche o resultado final na ordem original

        Isso é significativamente mais eficiente do que chamar embed()
        em loop — o SentenceTransformer processa batches em paralelo
        na GPU (se disponível) e com padding otimizado na CPU.

        Retorno: np.ndarray shape=(N, dims) dtype=float32
        """
        if not texts:
            return np.zeros(
                (0, self._model_config["dims"]), dtype=np.float32
            )

        model_id  = self._model_config["model_id"]
        dims      = self._model_config["dims"]
        n         = len(texts)
        result    = np.zeros((n, dims), dtype=np.float32)

        # separa hits de cache dos textos que precisam ser computados
        uncached_indices: list[int]  = []
        uncached_texts:   list[str]  = []

        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                # vetor zero para textos vazios — já está em result
                continue

            cached = self._cache.get(text, model_id)
            if cached is not None:
                result[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # computa apenas os textos sem cache
        if uncached_texts:
            model = self._get_model()
            t0    = time.time()

            new_vectors = model.encode(
                uncached_texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            ).astype(np.float32)

            elapsed = time.time() - t0
            logger.debug(
                f"[embeddings] {len(uncached_texts)} textos em {elapsed:.2f}s "
                f"({len(uncached_texts)/elapsed:.0f} textos/s)"
            )

            # preenche resultado e popula cache
            for idx, text, vector in zip(
                uncached_indices, uncached_texts, new_vectors
            ):
                result[idx] = vector
                self._cache.set(text, model_id, vector)

        return result

    # =========================================================================
    # Similaridade cosine
    # =========================================================================

    def cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Calcula similaridade cosine entre dois vetores normalizados.

        Com normalize=True no embed(), os vetores já têm norma 1.
        Nesse caso, cosine similarity = dot product — operação O(d) simples.

        Retorno: float em [-1.0, 1.0]
          1.0  = idênticos semanticamente
          0.0  = sem relação
         -1.0  = opostos (raro em texto natural)
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def cosine_similarity_matrix(
        self,
        vecs_a: np.ndarray,
        vecs_b: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula matriz de similaridade cosine entre dois conjuntos de vetores.

        Usado pelo similarity.py para comparar N claims contra M evidências
        em uma única operação matricial — muito mais eficiente que N×M chamadas.

        Entrada:
            vecs_a: shape (N, dims)
            vecs_b: shape (M, dims)
        Saída:
            shape (N, M) — sim[i][j] = similaridade entre vecs_a[i] e vecs_b[j]
        """
        # normaliza linhas para garantir cosine correto mesmo sem normalize=True
        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)

        # evita divisão por zero em vetores zero
        norms_a = np.where(norms_a == 0, 1e-9, norms_a)
        norms_b = np.where(norms_b == 0, 1e-9, norms_b)

        vecs_a_norm = vecs_a / norms_a
        vecs_b_norm = vecs_b / norms_b

        # produto matricial: (N, dims) × (dims, M) = (N, M)
        return np.dot(vecs_a_norm, vecs_b_norm.T).astype(np.float32)

    # =========================================================================
    # Utilitários
    # =========================================================================

    @property
    def dims(self) -> int:
        return self._model_config["dims"]

    @property
    def model_id(self) -> str:
        return self._model_config["model_id"]

    @property
    def max_tokens(self) -> int:
        return self._model_config["max_tokens"]

    def cache_stats(self) -> dict:
        return self._cache.stats()

    def warmup(self) -> None:
        """
        Pré-carrega o modelo com uma sentença dummy.
        Chamar no startup da API FastAPI evita o delay de 2-5s
        na primeira requisição real do usuário.
        """
        logger.info("[embeddings] warmup — pré-carregando modelo...")
        self.embed("aquecimento do modelo de embeddings")
        logger.info("[embeddings] warmup concluído")

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"EmbeddingModel("
            f"model='{self._model_name}', "
            f"dims={self.dims}, "
            f"loaded={loaded}, "
            f"cache={self._cache.stats()['size']} entradas"
            f")"
        )