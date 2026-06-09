# =============================================================================
# vector_store.py
# Gerencia o índice vetorial FAISS para busca semântica de documentos.
#
# Responsabilidades:
#   - Indexar documentos convertidos em embeddings pelo embeddings.py
#   - Persistir o índice em disco (data/vector_store/)
#   - Recuperar documentos semanticamente similares a uma query
#   - Sincronizar metadados no PostgreSQL (id, fonte, url, texto resumido)
#
# Usado por:
#   - retriever.py  → VectorStoreSource.search() (Camada 1 do RAG)
#   - pipeline.py   → indexação de novos documentos confiáveis
#
# Arquitetura híbrida (Figura 3 do TCC):
#   FAISS   → armazena vetores + id de referência
#   PostgreSQL → armazena metadados estruturados (texto, fonte, url, data)
#
# Entrada:  list[str] textos + metadados
# Saída:    list[dict] documentos rankeados por similaridade
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class Document:
    """
    Documento indexado na base vetorial.

    text:     texto completo — armazenado nos metadados, não no FAISS
    source:   nome do veículo (ex: "G1", "Folha de S.Paulo")
    url:      URL original do documento
    doc_id:   identificador único gerado na indexação
    chunk_id: posição do chunk dentro do documento original
              (documentos longos são divididos em chunks)
    published_at: data de publicação ISO 8601 quando disponível
    metadata: campos extras livres (categoria, autor, tags etc.)
    """
    text:         str
    source:       str
    url:          str
    doc_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id:     int = 0
    published_at: str | None = None
    metadata:     dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Resultado de uma busca vetorial com score de similaridade."""
    document:   Document
    score:      float        # similaridade cosine [0.0, 1.0]
    faiss_rank: int          # posição no ranking FAISS (0 = mais similar)


# =============================================================================
# VectorStore
# =============================================================================

class VectorStore:
    """
    Base vetorial FAISS com metadados persistidos em arquivo JSON.

    Por que JSON e não PostgreSQL direto aqui:
      O PostgreSQL será integrado na API FastAPI (camada de serviço),
      não dentro do VectorStore — mantém o módulo independente e
      testável sem banco de dados. O arquivo JSON de metadados é um
      espelho local que o PostgreSQL sincroniza via pipeline.py.

    Estrutura em disco:
      data/vector_store/
        ├── index.faiss       índice vetorial binário
        └── metadata.json     metadados dos documentos indexados

    Índice FAISS usado: IndexFlatIP (Inner Product)
      - Equivale a cosine similarity quando vetores são normalizados (L2=1)
      - embeddings.py já normaliza com normalize_embeddings=True
      - Mais rápido que IndexFlatL2 para vetores normalizados
      - Para > 100k documentos: migrar para IndexIVFFlat (aproximado)
    """

    # caminho padrão — pode ser sobrescrito via construtor ou .env
    DEFAULT_STORE_PATH = Path("data/vector_store")

    # limiar mínimo de similaridade para retornar um resultado
    MIN_SIMILARITY: float = 0.30

    # tamanho máximo de chunk em caracteres
    # documentos maiores são divididos para melhor granularidade semântica
    MAX_CHUNK_CHARS: int = 800

    # sobreposição entre chunks consecutivos (em caracteres)
    CHUNK_OVERLAP:   int = 100

    def __init__(
        self,
        store_path: str | Path | None = None,
        model_name: str = "multilingual-minilm",
    ):
        """
        Args:
            store_path: caminho para persistência do índice e metadados.
                        Se None, usa DEFAULT_STORE_PATH.
            model_name: modelo de embeddings — deve ser o mesmo usado
                        na indexação e na busca para garantir consistência.
        """
        self._path = Path(store_path or self.DEFAULT_STORE_PATH)
        self._path.mkdir(parents=True, exist_ok=True)

        self._index_path    = self._path / "index.faiss"
        self._metadata_path = self._path / "metadata.json"

        # lazy imports — evita erro de import se faiss não estiver instalado
        self._faiss  = None
        self._index  = None
        self._metadata: list[dict] = []   # espelho dos documentos indexados

        # modelo de embeddings — singleton compartilhado com outros módulos
        from pipeline.analysis.embeddings import EmbeddingModel
        self._embedding_model = EmbeddingModel(model_name)
        self._dims = self._embedding_model.dims

        # carrega índice existente se disponível
        self._load()

    # =========================================================================
    # FAISS — carregamento lazy
    # =========================================================================

    def _get_faiss(self):
        """
        Importa FAISS na primeira chamada.
        FAISS não é instalado por padrão — só falha quando realmente usado.
        """
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise ImportError(
                    "FAISS não instalado. Execute:\n"
                    "  pip install faiss-cpu   # CPU (recomendado para TCC)\n"
                    "  pip install faiss-gpu   # GPU (opcional)"
                )
        return self._faiss

    def _create_index(self):
        """
        Cria um índice FAISS vazio.

        IndexFlatIP: busca exata por Inner Product (= cosine com L2=1).
        Para escala maior (> 100k docs): IndexIVFFlat com nlist=100.
        """
        faiss = self._get_faiss()
        self._index = faiss.IndexFlatIP(self._dims)
        logger.info(f"[vector_store] índice criado: IndexFlatIP({self._dims}d)")

    # =========================================================================
    # Persistência
    # =========================================================================

    def _load(self) -> None:
        """
        Carrega índice FAISS e metadados do disco se existirem.
        Chamado no __init__ — o VectorStore fica pronto para uso imediato.
        """
        faiss = self._get_faiss()

        if self._index_path.exists() and self._metadata_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)

                logger.info(
                    f"[vector_store] índice carregado: "
                    f"{self._index.ntotal} vetores · "
                    f"{len(self._metadata)} documentos"
                )
            except Exception as e:
                logger.error(f"[vector_store] falha ao carregar índice: {e}")
                self._create_index()
                self._metadata = []
        else:
            # primeiro uso — cria índice vazio
            self._create_index()
            self._metadata = []
            logger.info("[vector_store] índice vazio criado (primeira execução)")

    def _save(self) -> None:
        """
        Persiste o índice FAISS e metadados em disco.
        Chamado após cada operação de indexação.
        """
        if self._index is None:
            return

        faiss = self._get_faiss()

        try:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)

            logger.debug(
                f"[vector_store] salvo: {self._index.ntotal} vetores"
            )
        except Exception as e:
            logger.error(f"[vector_store] falha ao salvar índice: {e}")
            raise

    # =========================================================================
    # Chunking de documentos longos
    # =========================================================================

    def _chunk_text(self, text: str) -> list[str]:
        """
        Divide texto longo em chunks com sobreposição.

        Por que chunkar:
          - Documentos longos produzem embeddings que "diluem" o significado
          - Chunks menores têm embeddings mais precisos semanticamente
          - Granularidade fina melhora o recall na busca
          - Consistente com a abordagem de segmentation.py

        Sobreposição (CHUNK_OVERLAP) garante que sentenças na fronteira
        entre dois chunks sejam cobertas por pelo menos um chunk completo.
        """
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        start  = 0

        while start < len(text):
            end = start + self.MAX_CHUNK_CHARS

            # tenta quebrar no espaço mais próximo para não cortar palavras
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # próximo chunk começa com sobreposição
            start = end - self.CHUNK_OVERLAP

        return chunks

    # =========================================================================
    # Indexação
    # =========================================================================

    def add_document(self, document: Document) -> list[str]:
        """
        Indexa um documento, dividindo em chunks se necessário.

        Retorna lista de doc_ids gerados (um por chunk).
        O doc_id base é preservado em todos os chunks para rastreabilidade.
        """
        if self._index is None:
            self._create_index()

        chunks    = self._chunk_text(document.text)
        doc_ids   = []

        # gera embeddings de todos os chunks em batch — mais eficiente
        vectors = self._embedding_model.embed_batch(
            chunks,
            normalize=True,   # obrigatório para IndexFlatIP = cosine similarity
        )

        for chunk_id, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            # FAISS armazena apenas o vetor — metadados ficam no JSON
            vector_2d = vector.reshape(1, -1).astype(np.float32)
            self._index.add(vector_2d)

            # posição no índice FAISS = len(metadata) antes do append
            faiss_id = len(self._metadata)

            chunk_doc_id = f"{document.doc_id}_{chunk_id}"
            doc_ids.append(chunk_doc_id)

            self._metadata.append({
                "faiss_id":     faiss_id,
                "doc_id":       chunk_doc_id,
                "base_doc_id":  document.doc_id,
                "chunk_id":     chunk_id,
                "text":         chunk_text,
                "source":       document.source,
                "url":          document.url,
                "published_at": document.published_at,
                "metadata":     document.metadata,
                "indexed_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        self._save()
        logger.info(
            f"[vector_store] indexado: '{document.source}' "
            f"→ {len(chunks)} chunk(s) · doc_id={document.doc_id}"
        )
        return doc_ids

    def add_documents(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Indexa múltiplos documentos em batch.
        Mais eficiente que chamar add_document() em loop porque
        o embed_batch() processa todos os chunks de uma vez.
        """
        if self._index is None:
            self._create_index()

        all_chunks:   list[str]      = []
        chunk_meta:   list[dict]     = []

        # expande todos os documentos em chunks
        for document in documents:
            chunks = self._chunk_text(document.text)
            for chunk_id, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_meta.append({
                    "document":  document,
                    "chunk_id":  chunk_id,
                    "chunk_text": chunk_text,
                })

        if not all_chunks:
            return []

        # embeddings de todos os chunks em um único batch
        vectors = self._embedding_model.embed_batch(
            all_chunks,
            normalize=True,
            show_progress=show_progress,
        )

        doc_ids = []
        vectors_to_add = vectors.astype(np.float32)
        base_faiss_id  = len(self._metadata)

        for i, (meta, vector) in enumerate(zip(chunk_meta, vectors_to_add)):
            document      = meta["document"]
            chunk_id      = meta["chunk_id"]
            chunk_text    = meta["chunk_text"]
            chunk_doc_id  = f"{document.doc_id}_{chunk_id}"

            doc_ids.append(chunk_doc_id)
            self._metadata.append({
                "faiss_id":     base_faiss_id + i,
                "doc_id":       chunk_doc_id,
                "base_doc_id":  document.doc_id,
                "chunk_id":     chunk_id,
                "text":         chunk_text,
                "source":       document.source,
                "url":          document.url,
                "published_at": document.published_at,
                "metadata":     document.metadata,
                "indexed_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        # adiciona todos os vetores ao FAISS em uma única operação
        self._index.add(vectors_to_add)
        self._save()

        logger.info(
            f"[vector_store] batch indexado: {len(documents)} documentos "
            f"→ {len(all_chunks)} chunks"
        )
        return doc_ids

    # =========================================================================
    # Busca
    # =========================================================================

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float | None = None,
        filter_source: str | None = None,
    ) -> list[dict]:
        """
        Busca os top_k documentos mais similares à query.

        Fluxo:
          1. Gera embedding da query (usa cache se já foi embedada)
          2. FAISS retorna os top_k índices + scores (Inner Product)
          3. Recupera metadados pelos índices
          4. Filtra por min_similarity e filter_source
          5. Retorna lista de dicts no formato esperado pelo retriever.py

        Args:
            query_text:     texto da query (claim.normalized do claim_detector)
            top_k:          número máximo de resultados
            min_similarity: threshold mínimo (sobrescreve MIN_SIMILARITY se fornecido)
            filter_source:  retorna apenas documentos desta fonte (ex: "G1")

        Retorno (formato consumido por VectorStoreSource no retriever.py):
            list[dict] com keys: text, source, url, score, published_at, metadata
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("[vector_store] índice vazio — nenhum resultado")
            return []

        threshold = min_similarity if min_similarity is not None else self.MIN_SIMILARITY

        # embedding da query — usa cache do EmbeddingModel
        query_vector = self._embedding_model.embed(
            query_text,
            normalize=True,
        ).reshape(1, -1).astype(np.float32)

        # busca FAISS — retorna (scores, indices) shape=(1, top_k)
        # busca mais resultados que top_k para compensar filtragem posterior
        search_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(query_vector, search_k)

        results = []
        seen_base_doc_ids: set[str] = set()

        for score, idx in zip(scores[0], indices[0]):
            # FAISS retorna -1 para posições não preenchidas
            if idx == -1:
                continue

            # score do IndexFlatIP com vetores normalizados = cosine similarity
            similarity = float(score)

            if similarity < threshold:
                continue

            meta = self._metadata[idx]

            # filtra por fonte se especificado
            if filter_source and meta["source"] != filter_source:
                continue

            # deduplica por documento base — evita retornar múltiplos
            # chunks do mesmo artigo (prefere o chunk com maior score)
            base_id = meta["base_doc_id"]
            if base_id in seen_base_doc_ids:
                continue
            seen_base_doc_ids.add(base_id)

            results.append({
                "text":         meta["text"],
                "source":       meta["source"],
                "url":          meta["url"],
                "score":        round(similarity, 4),
                "published_at": meta.get("published_at"),
                "metadata":     meta.get("metadata", {}),
                "doc_id":       meta["doc_id"],
                "faiss_rank":   len(results),
            })

            if len(results) >= top_k:
                break

        logger.debug(
            f"[vector_store] query: '{query_text[:60]}...' "
            f"→ {len(results)} resultados (threshold={threshold})"
        )
        return results

    # =========================================================================
    # Utilitários
    # =========================================================================

    def remove_document(self, base_doc_id: str) -> int:
        """
        Remove todos os chunks de um documento pelo base_doc_id.

        FAISS não suporta remoção nativa no IndexFlatIP — a estratégia
        é marcar como removido nos metadados e reconstruir o índice
        periodicamente. Para o TCC, remoção simples via reconstrução.

        Retorna: número de chunks removidos.
        """
        # identifica chunks a remover
        to_remove = {
            i for i, m in enumerate(self._metadata)
            if m["base_doc_id"] == base_doc_id
        }

        if not to_remove:
            logger.warning(f"[vector_store] doc_id não encontrado: {base_doc_id}")
            return 0

        # filtra metadados mantendo apenas os não removidos
        surviving = [
            (i, m) for i, m in enumerate(self._metadata)
            if i not in to_remove
        ]

        if not surviving:
            self._create_index()
            self._metadata = []
            self._save()
            return len(to_remove)

        # reconstrói o índice com os vetores sobreviventes
        # (necessário pois FAISS não suporta remoção direta no IndexFlatIP)
        surviving_indices = [i for i, _ in surviving]
        surviving_meta    = [m for _, m in surviving]

        faiss = self._get_faiss()

        # extrai vetores dos índices sobreviventes
        # reconstruct_batch recupera os vetores originais do índice
        surviving_vectors = np.vstack([
            self._index.reconstruct(i).reshape(1, -1)
            for i in surviving_indices
        ]).astype(np.float32)

        # recria índice com vetores sobreviventes
        self._create_index()
        self._index.add(surviving_vectors)

        # atualiza faiss_id nos metadados para refletir nova posição
        for new_id, meta in enumerate(surviving_meta):
            meta["faiss_id"] = new_id

        self._metadata = surviving_meta
        self._save()

        removed_count = len(to_remove)
        logger.info(
            f"[vector_store] removido: {base_doc_id} "
            f"({removed_count} chunk(s))"
        )
        return removed_count

    @property
    def document_count(self) -> int:
        """Número de chunks indexados (não de documentos únicos)."""
        return self._index.ntotal if self._index else 0

    @property
    def unique_document_count(self) -> int:
        """Número de documentos únicos (base_doc_id distintos)."""
        return len({m["base_doc_id"] for m in self._metadata})

    def stats(self) -> dict:
        """Estatísticas do índice para log e monitoramento."""
        sources: dict[str, int] = {}
        for m in self._metadata:
            sources[m["source"]] = sources.get(m["source"], 0) + 1

        return {
            "total_chunks":     self.document_count,
            "unique_documents": self.unique_document_count,
            "embedding_dims":   self._dims,
            "index_path":       str(self._index_path),
            "index_size_mb":    round(
                self._index_path.stat().st_size / 1_048_576, 2
            ) if self._index_path.exists() else 0,
            "sources":          sources,
        }

    def __repr__(self) -> str:
        return (
            f"VectorStore("
            f"chunks={self.document_count}, "
            f"docs={self.unique_document_count}, "
            f"dims={self._dims}, "
            f"path='{self._path}'"
            f")"
        )