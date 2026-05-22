import re
import unicodedata
from functools import lru_cache
from typing import Literal


class TextNormalizer:

    _nlp = None

    # stopwords extras do domínio jornalístico brasileiro
    # complementam as do spaCy sem substituir
    _JOURNALISTIC_STOPWORDS = {
        # verbos de citação — frequentíssimos mas sem valor semântico
        "afirmou", "disse", "declarou", "informou", "relatou",
        "explicou", "destacou", "ressaltou", "apontou", "acrescentou",
        "afirma", "diz", "declara", "informa", "relata", "aponta",
        # marcadores de fonte genéricos
        "segundo", "conforme", "de acordo", "portal", "redação",
        "agência", "correspondente", "enviado especial",
        # portais e veículos (não agregam ao tema da notícia)
        "g1", "uol", "folha", "globo", "estadão", "veja",
        "r7", "record", "sbt", "band",
        # expressões de tempo genéricas
        "hoje", "ontem", "semana", "mês", "ano", "passado",
        "próximo", "anterior", "recente", "recentemente",
    }

    # abreviações comuns em notícias brasileiras → forma expandida
    # normaliza antes da lematização para o spaCy reconhecer melhor
    _ABBREVIATIONS = {
        r"\bsp\b": "São Paulo",
        r"\brj\b": "Rio de Janeiro",
        r"\bdf\b": "Distrito Federal",
        r"\bpm\b": "Polícia Militar",
        r"\bpc\b": "Polícia Civil",
        r"\bmp\b": "Ministério Público",
        r"\bstf\b": "Supremo Tribunal Federal",
        r"\bstj\b": "Superior Tribunal de Justiça",
        r"\btcu\b": "Tribunal de Contas da União",
        r"\bpib\b": "Produto Interno Bruto",
        r"\bipca\b": "índice de preços ao consumidor",
        r"\bsus\b": "Sistema Único de Saúde",
        r"\bonu\b": "Organização das Nações Unidas",
        r"\beua\b": "Estados Unidos",
        r"\bue\b": "União Europeia",
    }

    # compila os padrões de abreviação uma única vez
    _ABBREV_PATTERNS = [
        (re.compile(pat, re.IGNORECASE), expansion)
        for pat, expansion in _ABBREVIATIONS.items()
    ]

    # ── carregamento do modelo ────────────────────────────────────────────────

    @classmethod
    def _get_nlp(cls):
        """
        Lazy loading do modelo spaCy.
        Desabilita componentes desnecessários para ganhar velocidade:
        - ner: reconhecimento de entidades (feito em módulo próprio)
        - parser: dependência sintática (não usada na normalização)
        Mantém: tokenizer, tagger (POS), lemmatizer.
        """
        if cls._nlp is None:
            try:
                import spacy
                cls._nlp = spacy.load(
                    "pt_core_news_sm",
                    disable=["ner", "parser"]
                )
                cls._nlp.max_length = 2_000_000

                # adiciona stopwords do domínio ao vocabulário do spaCy
                for word in cls._JOURNALISTIC_STOPWORDS:
                    cls._nlp.vocab[word].is_stop = True

            except OSError:
                raise ImportError(
                    "Modelo spaCy não encontrado. Execute:\n"
                    "  python -m spacy download pt_core_news_sm"
                )
        return cls._nlp

    # ── transformações individuais ────────────────────────────────────────────

    @staticmethod
    def _to_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _remove_accents(text: str) -> str:
        """
        NFD → remove marcas diacríticas (categoria Mn).
        Preserva caracteres como ç (c + cedilla decompostos → recompostos via NFC).
        Resultado: 'São Paulo' → 'Sao Paulo', 'ação' → 'acao'.
        """
        nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", stripped)

    @classmethod
    def _expand_abbreviations(cls, text: str) -> str:
        """
        Expande siglas antes da lematização para o modelo reconhecer
        a forma completa e lematizar corretamente.
        """
        for pattern, expansion in cls._ABBREV_PATTERNS:
            text = pattern.sub(expansion, text)
        return text

    @staticmethod
    def _process_tokens(
        doc,
        remove_stopwords: bool,
        lemmatize: bool,
        min_token_length: int,
        keep_entities: bool,
    ) -> list[str]:
        """
        Processa tokens do spaCy aplicando filtros configuráveis.

        keep_entities: se True, preserva tokens que seriam stopwords
        mas fazem parte de entidades nomeadas (ex: "de" em "Banco de Brasil"
        não deveria ser removido num contexto de NER).
        """
        tokens = []
        for token in doc:
            # sempre descarta pontuação, espaços e números puros
            if token.is_punct or token.is_space:
                continue
            if token.like_num:
                continue

            form = token.lemma_ if lemmatize else token.text

            if len(form) < min_token_length:
                continue

            # stopword: verifica se token deve ser mantido por contexto de entidade
            if remove_stopwords and token.is_stop:
                if keep_entities and token.ent_type_:
                    pass  # mantém — faz parte de entidade nomeada
                else:
                    continue

            tokens.append(form)

        return tokens

    # ── perfis de normalização ────────────────────────────────────────────────

    @classmethod
    def normalize(
        cls,
        text: str,
        lowercase: bool = True,
        remove_accents: bool = True,
        expand_abbreviations: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        keep_entities: bool = False,
        min_token_length: int = 2,
    ) -> str:
        """
        Normalização completa — perfil padrão para TF-IDF, embeddings clássicos,
        busca semântica e similarity.py.

        Parâmetros booleanos permitem que módulos downstream configurem
        o nível exato de normalização sem criar subclasses.
        """
        if not text or not text.strip():
            return ""

        if expand_abbreviations:
            text = cls._expand_abbreviations(text)

        if lowercase:
            text = cls._to_lowercase(text)

        if remove_accents:
            text = cls._remove_accents(text)

        nlp = cls._get_nlp()
        doc = nlp(text)

        tokens = cls._process_tokens(
            doc,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            min_token_length=min_token_length,
            keep_entities=keep_entities,
        )

        return " ".join(tokens)

    @classmethod
    def normalize_for_bert(cls, text: str) -> str:
        """
        Perfil para BERTimbau e modelos baseados em Transformers.

        BERTimbau foi treinado em português com acentos, caixa mista
        e stopwords — alterá-los degrada a representação vetorial.
        Faz apenas expansão de abreviações para ajudar o tokenizador WordPiece.
        """
        if not text or not text.strip():
            return ""

        # só expande abreviações — o resto o BERT lida internamente
        text = cls._expand_abbreviations(text)

        # remove espaços duplos que possam ter sobrado
        text = re.sub(r" {2,}", " ", text).strip()

        return text

    @classmethod
    def normalize_for_tfidf(cls, text: str) -> str:
        """
        Perfil para TF-IDF e modelos bag-of-words clássicos.
        Normalização máxima: tudo em minúscula, sem acento, lematizado, sem stopwords.
        """
        return cls.normalize(
            text,
            lowercase=True,
            remove_accents=True,
            expand_abbreviations=True,
            remove_stopwords=True,
            lemmatize=True,
            min_token_length=3,  # mais restrito — evita ruído em vocabulários grandes
        )

    @classmethod
    def normalize_for_similarity(cls, text: str) -> str:
        """
        Perfil para similarity.py e stance_model.py.
        Mantém stopwords e não lematiza — preserva estrutura frásica
        para que métricas como cosine/BM25 capturem proximidade sintática.
        """
        return cls.normalize(
            text,
            lowercase=True,
            remove_accents=True,
            expand_abbreviations=True,
            remove_stopwords=False,
            lemmatize=False,
            min_token_length=2,
        )

    # ── processamento em lote ─────────────────────────────────────────────────

    @classmethod
    def normalize_blocks(
        cls,
        blocks: list[str],
        profile: Literal["default", "bert", "tfidf", "similarity"] = "default",
    ) -> list[str]:
        """
        Normaliza lista de blocos usando o perfil especificado.
        Descarta blocos que ficam vazios após normalização.

        Args:
            blocks:  lista de blocos limpos vindos do cleaner
            profile: qual perfil de normalização aplicar
        """
        profile_map = {
            "default":    cls.normalize,
            "bert":       cls.normalize_for_bert,
            "tfidf":      cls.normalize_for_tfidf,
            "similarity": cls.normalize_for_similarity,
        }

        fn = profile_map.get(profile, cls.normalize)
        result = []

        for block in blocks:
            norm = fn(block)
            if norm:
                result.append(norm)

        return result