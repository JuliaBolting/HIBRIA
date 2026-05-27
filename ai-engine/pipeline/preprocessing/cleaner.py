import re
import unicodedata


class TextCleaner:

    # ── padrões compilados uma vez na classe ────────────────────────────────

    _URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
    _MENTION_RE = re.compile(r"[@#]\w+")

    # números de telefone brasileiros: (11) 99999-9999, +55 11 99999-9999 etc.
    _PHONE_RE = re.compile(
        r"(\+?55\s?)?(\(?\d{2}\)?\s?)(\d{4,5}[-\s]?\d{4})"
    )

    # CPF, CNPJ, CEP — identificadores que não agregam semântica noticiosa
    _DOCUMENT_RE = re.compile(
        r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"   # CPF
        r"|\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"  # CNPJ
        r"|\b\d{5}-?\d{3}\b"                   # CEP
    )

    # sequências longas de dígitos soltos (IDs, timestamps, hashes em URLs copiadas)
    _LONG_NUMBER_RE = re.compile(r"\b\d{7,}\b")

    # pontuação repetida
    _REPEATED_PUNCT_RE = re.compile(r"([!?.\-_*=~])\1{2,}")

    # múltiplos espaços/tabs numa linha
    _WHITESPACE_RE = re.compile(r"[ \t]+")

    # mais de 2 quebras de linha seguidas
    _NEWLINE_RE = re.compile(r"\n{3,}")

    # caracteres de controle e zero-width
    _CONTROL_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
        r"\u200b-\u200f\u202a-\u202e\ufeff]"
    )

    # artefatos comuns do Playwright/SPAs: JSON inline, props de componente
    _JSON_BLOB_RE = re.compile(r"\{[^{}]{80,}\}", re.DOTALL)
    _HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]{2,6};|&#\d{1,5};|&#x[0-9a-fA-F]{1,4};")

    # texto que vem colado sem espaço por falha de extração de SPA
    # ex: "PalavraOutraPalavra" → detecta camelCase não intencional
    _CAMEL_GLUE_RE = re.compile(r"([a-záàãâéêíóôõúç])([A-ZÁÀÃÂÉÊÍÓÔÕÚÇ])")

    # ruído de UI brasileiro — expandido para cobrir mais portais
    _NOISE_PHRASES_RE = re.compile(
        r"(aceitar?\s+(todos\s+os\s+)?cookie"
        r"|política\s+de\s+privacidade"
        r"|termos\s+de\s+uso"
        r"|assine\s+(agora|já|o\s+\w+)"
        r"|leia\s+(mais|também)"
        r"|continue\s+lendo"
        r"|veja\s+(também|mais)"
        r"|publicidade|anúncio|patrocinado"
        r"|compartilhe\s+(esta?\s+)?(notícia|matéria|artigo)"
        r"|coment[aá](r|rios?)"
        r"|carregando\.\.\."
        r"|voltar\s+ao\s+topo"
        r"|ouça\s+(esta?\s+)?notícia"
        r"|tempo\s+de\s+leitura"
        r"|participe\s+do\s+canal"
        r"|no\s+whatsapp"
        r"|veja\s+mais\s+not[ií]cias"
        r"|not[ií]cias\s+da\s+regi[aã]o"
        r"|v[ií]deos?:\s*assista"
        r"|assista\s+[àa]s?\s+reportagens"
        r"|minutos?\s+de\s+leitura)",
        re.IGNORECASE
    )

    # ── métodos de remoção ───────────────────────────────────────────────────

    @staticmethod
    def _remove_emojis(text: str) -> str:
        """
        Remove emojis por categoria Unicode.
        Cobre emojis base (So), modificadores (Sk) e pares surrogate (Cs).
        Mantém símbolos tipográficos úteis como © ® ™.
        """
        KEEP_SYMBOLS = {"©", "®", "™", "°", "§", "%", "&"}
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat in ("So", "Cs", "Sk") and ch not in KEEP_SYMBOLS:
                result.append(" ")  # substitui por espaço, não remove — preserva estrutura
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _decode_html_entities(text: str) -> str:
        """
        Decodifica entidades HTML residuais que o BeautifulSoup/Playwright
        às vezes deixa passar (&amp; &nbsp; &#8211; etc).
        """
        import html
        return html.unescape(text)

    @staticmethod
    def _fix_camel_glue(text: str) -> str:
        """
        Insere espaço em colagens camelCase acidentais vindas de SPAs.
        'PrefeitoraAssina' → 'Prefeitoraa ssina' (melhor que junto)
        Só aplica quando NÃO parece sigla (evita 'SãoPaulo' → 'São Paulo' OK,
        mas 'iPhone' não deveria ser tocado — aceita falsos positivos mínimos).
        """
        return TextCleaner._CAMEL_GLUE_RE.sub(r"\1 \2", text)

    # ── pipeline de limpeza ──────────────────────────────────────────────────

    @classmethod
    def clean(cls, text: str, source: str = "static") -> str:
        """
        Limpa texto bruto extraído da web.

        Args:
            text:   texto bruto do extractor
            source: "static" | "playwright"
                    Playwright tende a trazer mais artefatos JS,
                    ativando etapas extras de limpeza.
        """
        if not text or not text.strip():
            return ""

        # 1. decodifica entidades HTML antes de qualquer coisa
        text = cls._decode_html_entities(text)

        # 2. caracteres de controle e zero-width
        text = cls._CONTROL_RE.sub("", text)

        # 3. artefatos específicos de SPA/Playwright
        if source == "playwright":
            text = cls._JSON_BLOB_RE.sub(" ", text)        # blobs JSON inline
            text = cls._fix_camel_glue(text)               # colagens camelCase

        # 4. entidades HTML residuais (segunda passagem após unescape)
        text = cls._HTML_ENTITY_RE.sub(" ", text)

        # 5. URLs, emails, menções
        text = cls._URL_RE.sub(" ", text)
        text = cls._EMAIL_RE.sub(" ", text)
        text = cls._MENTION_RE.sub(" ", text)

        # 6. identificadores numéricos (telefone, CPF, CNPJ, CEP, IDs longos)
        text = cls._PHONE_RE.sub(" ", text)
        text = cls._DOCUMENT_RE.sub(" ", text)
        text = cls._LONG_NUMBER_RE.sub(" ", text)

        # 7. emojis e símbolos decorativos
        text = cls._remove_emojis(text)

        # 8. frases de ruído de UI
        text = cls._NOISE_PHRASES_RE.sub(" ", text)

        # 9. pontuação repetida → mantém uma ocorrência
        text = cls._REPEATED_PUNCT_RE.sub(r"\1", text)

        # 10. normalização de espaços (preserva parágrafos)
        text = cls._WHITESPACE_RE.sub(" ", text)
        text = cls._NEWLINE_RE.sub("\n\n", text)

        # 11. remove linhas que ficaram com menos de 15 chars após limpeza
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) >= 15 or stripped == "":
                lines.append(stripped)
        text = "\n".join(lines)

        return text.strip()

    @classmethod
    def clean_blocks(cls, blocks: list[str], source: str = "static") -> list[str]:
        """
        Limpa lista de blocos do extractor, descartando os que ficam vazios.
        Repassa o `source` para ativar limpeza extra em conteúdo de Playwright.
        """
        cleaned = []
        for block in blocks:
            result = cls.clean(block, source=source)
            if result:
                cleaned.append(result)
        return cleaned