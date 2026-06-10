# =============================================================================
# Indexação automática de notícias aprovadas pelo sistema.
#
# Depois que o pipeline calcula score_final e label_final, este módulo decide
# se a notícia analisada pode ser adicionada ao FAISS como evidência futura.
#
# Importante:
#   - Não marca como trusted_news.
#   - Notícias aprovadas pelo sistema entram como analyzed_news.
#   - A notícia só é indexada se atingir o limite mínimo de confiança.
# =============================================================================

from __future__ import annotations

from urllib.parse import urlparse

from pipeline.retrieval.vector_store import VectorStore, Document


class AutoIndexer:
    """
    Indexa automaticamente notícias analisadas e aprovadas pelo sistema.

    Essa etapa permite que notícias com alto índice de confiabilidade
    passem a compor a base vetorial usada pelo RAG em análises futuras.
    """

    MIN_SCORE_TO_INDEX = 75.0

    @staticmethod
    def _extract_domain(url: str) -> str:
        domain = urlparse(url or "").netloc.lower().strip()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    @staticmethod
    def _already_indexed(vector_store: VectorStore, url: str) -> bool:
        """
        Verifica se a URL já existe nos metadados locais do FAISS.
        Evita indexar a mesma notícia várias vezes.
        """
        metadata = getattr(vector_store, "_metadata", [])

        for item in metadata:
            if item.get("url") == url:
                return True

        return False

    @classmethod
    def should_index(cls, result) -> tuple[bool, str]:
        """
        Decide se a notícia deve ser indexada.
        Retorna:
            (True, motivo)  ou  (False, motivo)
        """
        if not result.url:
            return False, "URL ausente."

        if not result.blocks_clean:
            return False, "Texto limpo ausente."

        if result.score_final is None:
            return False, "Score final ainda não calculado."

        if result.score_final < cls.MIN_SCORE_TO_INDEX:
            return (
                False,
                f"Score final abaixo do limite mínimo ({result.score_final} < {cls.MIN_SCORE_TO_INDEX}).",
            )

        if result.label_final != "confiável":
            return False, f"Rótulo final não aprovado para indexação: {result.label_final}."

        return True, "Notícia aprovada para indexação."

    @classmethod
    def index_result(cls, result, vector_store: VectorStore | None = None) -> dict:
        """
        Indexa a notícia analisada no FAISS como analyzed_news.
        """
        can_index, reason = cls.should_index(result)

        if not can_index:
            return {
                "indexed": False,
                "reason": reason,
                "doc_ids": [],
            }

        vector_store = vector_store or VectorStore()

        if cls._already_indexed(vector_store, result.url):
            return {
                "indexed": False,
                "reason": "Notícia já estava indexada no FAISS.",
                "doc_ids": [],
            }

        domain = cls._extract_domain(result.url)
        text = "\n".join(result.blocks_clean).strip()

        document = Document(
            text=text,
            source=domain or "noticia_analisada",
            url=result.url,
            published_at=None,
            metadata={
                "title": result.title,
                "description": result.description,
                "domain": domain,
                "source_type": "analyzed_news",
                "trusted_source": False,
                "approved_by_system": True,
                "confidence_score": result.score_final,
                "label_final": result.label_final,
                "collection_method": "auto_index_after_analysis",
            },
        )

        doc_ids = vector_store.add_document(document)

        return {
            "indexed": True,
            "reason": "Notícia indexada automaticamente como analyzed_news.",
            "doc_ids": doc_ids,
            "source_type": "analyzed_news",
            "domain": domain,
        }