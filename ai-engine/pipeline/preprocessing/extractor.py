# =============================================================================
# extractor.py
# Responsável pela extração do conteúdo textual de páginas web.
#
# Funciona em duas estratégias complementares:
#   1. Requisição HTTP direta (requests + BeautifulSoup) — leve e rápida,
#      funciona para a maioria dos portais que servem HTML completo no servidor.
#   2. Renderização com Playwright (Chromium headless) — acionada como fallback
#      quando a estratégia estática retorna uma SPA vazia ou falha por bloqueio.
#
# O resultado é um dicionário padronizado consumido pelo pipeline.py,
# que repassa os dados para cleaner.py e normalization.py.
# =============================================================================

import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# =============================================================================
# Exceções customizadas
# Separar ExtractionError (fatal) de ExtractionWarning (não-fatal) permite que
# o pipeline.py decida se interrompe o fluxo ou apenas registra o aviso.
# =============================================================================

class ExtractionError(Exception):
    """Erro fatal — extração impossível. O pipeline deve interromper o fluxo."""
    pass


class ExtractionWarning(Exception):
    """
    Conteúdo parcial extraído.
    Sinaliza limitações (paywall, fallback para Playwright) sem
    interromper o pipeline — a análise continua com o que foi obtido.
    """
    pass


# =============================================================================
# TextExtractor
# Classe principal de extração. Todos os métodos são estáticos ou de classe
# para evitar estado compartilhado entre requisições — importante quando o
# pipeline processar múltiplas URLs em sequência.
# =============================================================================

class TextExtractor:

    # -------------------------------------------------------------------------
    # Headers HTTP que simulam um navegador real.
    # Muitos portais bloqueiam requisições sem User-Agent ou com o padrão
    # do requests ("python-requests/x.x"). O conjunto completo de headers
    # reduz a chance de bloqueio por fingerprinting básico.
    # -------------------------------------------------------------------------
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",                        # Do Not Track — sinal de usuário real
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",  # prefere HTTPS quando disponível
    }

    # -------------------------------------------------------------------------
    # Tags HTML que nunca contêm texto útil para análise de notícias.
    # São removidas antes de qualquer extração de texto para reduzir ruído
    # e evitar que scripts/estilos inline sejam capturados como conteúdo.
    # -------------------------------------------------------------------------
    TAGS_TO_REMOVE = [
        "script",       # código JavaScript — seria capturado como texto
        "style",        # CSS inline
        "iframe",       # conteúdo externo embutido (anúncios, vídeos)
        "footer",       # rodapé: links legais, copyright, redes sociais
        "nav",          # menus de navegação
        "header",       # cabeçalho do site (logo, busca, menu principal)
        "aside",        # conteúdo lateral (widgets, publicidade)
        "figure",       # imagens e suas legendas
        "figcaption",   # legendas de imagem
        "form",         # formulários (busca, login, newsletter)
        "button",       # botões de UI
        "input",        # campos de formulário
        "select",       # dropdowns
        "svg",          # ícones vetoriais
        "canvas",       # gráficos e animações
        "video",        # players de vídeo
        "audio",        # players de áudio
        "noscript",     # conteúdo alternativo para JS desativado (geralmente anúncios)
        "template",     # templates HTML não renderizados
    ]

    # -------------------------------------------------------------------------
    # Seletores CSS de elementos de ruído que escapam da remoção por tag.
    # Usa substrings de class/id para cobrir variações de nomenclatura.
    #
    # IMPORTANTE: seletores como [class*='ad'] são perigosos — pegam classes
    # como "gradient", "upload", "read" que contêm "ad" mas não são anúncios.
    # Por isso usamos prefixo/sufixo mais específicos: 'ad-' e '-ad'.
    # -------------------------------------------------------------------------
    SELECTORS_TO_REMOVE = [
        "[class*='ad-']", "[class*='-ad']", "[id*='google-ad']",  # publicidade
        "[class*='banner']",        # banners promocionais
        "[class*='popup']",         # popups de assinatura, cookies etc.
        "[class*='cookie']",        # avisos de LGPD/GDPR
        "[class*='newsletter']",    # formulários de newsletter
        "[class*='paywall']",       # blocos de paywall
        "[class*='subscription']",  # blocos de assinatura
        "[class*='related-']",      # seção "leia também" / matérias relacionadas
        "[class*='recommended']",   # recomendações algorítmicas
        "[class*='share-']",        # botões de compartilhamento
        "[class*='social-share']",  # ícones de redes sociais
        "[class*='comment']",       # seção de comentários
        "[class*='sidebar']",       # barra lateral
        ".wall.protected-content",  # paywall específico dos portais Globo (G1, GE)
    ]

    # -------------------------------------------------------------------------
    # Sinais no HTML que indicam que a página é uma SPA (Single Page Application)
    # e precisa de renderização JavaScript para exibir o conteúdo.
    # Usado em conjunto com a checagem de conteúdo curto — um sinal JS sozinho
    # não é suficiente, pois muitos portais usam frameworks mas servem SSR.
    # -------------------------------------------------------------------------
    JS_SIGNALS = [
        "__NEXT_DATA__",        # Next.js (React SSR/SSG)
        "__NUXT__",             # Nuxt.js (Vue SSR)
        "window.__STATE__",     # padrão Redux de hidratação do estado
        "ng-version",           # Angular
        '<div id="app">',       # Vue SPA sem SSR
        '<div id="root">',      # React SPA sem SSR
    ]

    # -------------------------------------------------------------------------
    # Frases que indicam presença de paywall.
    # A detecção é feita no HTML bruto (antes da limpeza) para capturar
    # tanto texto visível quanto metadados e atributos ocultos.
    # -------------------------------------------------------------------------
    PAYWALL_SIGNALS = [
        "assine para continuar",
        "conteúdo exclusivo para assinantes",
        "subscribe to continue",
        "this content is for subscribers",
        "acesso restrito",
        "conteúdo bloqueado",
    ]

    # -------------------------------------------------------------------------
    # Domínios que não servem conteúdo legível via requisição HTTP direta.
    # Redes sociais retornam HTML mínimo ou redirecionam para autenticação —
    # tentar extrair deles geraria erros confusos ou conteúdo inútil.
    # -------------------------------------------------------------------------
    UNSUPPORTED_DOMAINS = {
        "twitter.com", "x.com", "instagram.com",
        "facebook.com", "tiktok.com", "linkedin.com",
    }

    # -------------------------------------------------------------------------
    # Seletores CSS ordenados por especificidade e confiabilidade.
    # Usados na camada 2 do _extract_blocks quando os roots semânticos
    # (article, main, itemprop) não trouxeram conteúdo suficiente.
    #
    # A ordem importa: o primeiro seletor que retornar >= 300 chars é usado.
    # Seletores mais específicos (de portais conhecidos) vêm antes dos genéricos
    # para evitar capturar menus ou sidebars em sites sem estrutura semântica.
    # -------------------------------------------------------------------------
    PORTAL_PARAGRAPH_SELECTORS = [
        "p.content-text__container",    # G1, GE e demais portais Globo
        "p.article__text",              # Folha de S.Paulo
        "[class*='article-body'] p",    # padrão genérico de portais de notícia
        "[class*='story-body'] p",      # Reuters, AP (versões em português)
        "[class*='mat-body'] p",        # Estadão
        "[class*='news-body'] p",       # portais regionais brasileiros
        "[class*='entry-content'] p",   # WordPress — usado por milhares de blogs
        "[class*='post-content'] p",    # tema genérico de blog/portal menor
        "[itemprop='articleBody'] p",   # schema.org — fallback semântico amplo
    ]

    # =========================================================================
    # Helpers de validação e detecção
    # Métodos internos que respondem perguntas simples sobre a URL ou o HTML.
    # =========================================================================

    @staticmethod
    def _validate_url(url: str) -> None:
        """
        Verifica se a URL tem esquema HTTP ou HTTPS.
        Rejeita file://, ftp://, javascript: e outros esquemas não suportados
        antes de fazer qualquer requisição de rede.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ExtractionError(f"Esquema inválido: '{url}'")

    @staticmethod
    def _is_unsupported_domain(url: str) -> bool:
        """
        Verifica se o domínio está na lista de domínios não suportados.
        Usa endswith para cobrir subdomínios (ex: mobile.twitter.com).
        """
        host = urlparse(url).hostname or ""
        return any(host.endswith(d) for d in TextExtractor.UNSUPPORTED_DOMAINS)

    @staticmethod
    def _detect_paywall(soup: BeautifulSoup, raw_html: str) -> bool:
        """
        Busca sinais de paywall no HTML bruto (em minúsculas).
        A busca é feita no HTML completo e não no texto visível porque
        alguns portais ocultam o bloqueio em atributos data-* ou comentários.
        """
        text_lower = raw_html.lower()
        return any(signal in text_lower for signal in TextExtractor.PAYWALL_SIGNALS)

    @staticmethod
    def _needs_js_render(html: str, blocks: list[str]) -> bool:
        """
        Determina se a página precisa de renderização JavaScript.

        Dois critérios precisam ser verdadeiros simultaneamente:
          1. HTML contém sinal de framework SPA (Next.js, Nuxt, Angular etc.)
          2. Conteúdo extraído tem menos de 300 caracteres no total

        Exigir ambos evita acionar o Playwright em portais que usam frameworks
        modernos mas servem HTML completo via SSR (Server-Side Rendering).
        """
        has_js_signal = any(signal in html for signal in TextExtractor.JS_SIGNALS)
        content_too_short = sum(len(b) for b in blocks) < 300
        return has_js_signal and content_too_short

    # =========================================================================
    # Estratégia 1: requests + BeautifulSoup
    # Abordagem padrão — leve, rápida, sem dependências externas pesadas.
    # Funciona para portais que renderizam o HTML no servidor (SSR).
    # =========================================================================

    @staticmethod
    def _fetch_static(url: str, retries: int = 3) -> requests.Response:
        """
        Realiza requisição HTTP com retry e backoff exponencial.

        Tentativas: até 3 por padrão.
        Espera entre tentativas: 1s → 2s → 4s (2^attempt).

        Erros recuperáveis (Timeout, ConnectionError, HTTP 403/429) acionam
        retry. Outros erros HTTP (404, 500 etc.) são fatais e sobem direto.

        HTTP 403 e 429 são tratados como recuperáveis porque muitos portais
        os retornam temporariamente para bots — uma segunda tentativa
        com backoff frequentemente resolve o bloqueio de rate limiting.
        """
        last_exc = None

        for attempt in range(retries):
            try:
                response = requests.get(
                    url,
                    headers=TextExtractor.HEADERS,
                    timeout=15,           # 15s: equilibra paciência e UX
                    allow_redirects=True, # segue redirecionamentos (HTTP → HTTPS)
                )
                # lança HTTPError para status 4xx e 5xx
                response.raise_for_status()
                return response

            except requests.exceptions.Timeout:
                last_exc = ExtractionError(
                    f"Timeout na tentativa {attempt + 1}: {url}"
                )
            except requests.exceptions.ConnectionError:
                # DNS não resolveu, conexão recusada, rede indisponível
                last_exc = ExtractionError(
                    f"Falha de conexão na tentativa {attempt + 1}: {url}"
                )
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status in (403, 429) and attempt < retries - 1:
                    # possível rate limiting ou bloqueio temporário de bot
                    last_exc = ExtractionError(
                        f"HTTP {status} — possível bloqueio de bot"
                    )
                else:
                    # erros definitivos: 404, 500, 503 etc.
                    raise ExtractionError(f"HTTP {status}: {url}")

            # backoff exponencial: 1s, 2s, 4s
            time.sleep(2 ** attempt)

        raise last_exc

    # =========================================================================
    # Estratégia 2: Playwright (Chromium headless)
    # Fallback para SPAs e páginas que dependem de JavaScript para renderizar
    # o conteúdo. Mais lento (~3-8s) mas cobre casos que requests não alcança.
    # =========================================================================

    @staticmethod
    def _fetch_with_playwright(url: str) -> str:
        """
        Renderiza a página com Chromium headless via Playwright.

        Fluxo:
          1. Abre o navegador e bloqueia recursos desnecessários (imagens, fontes,
             mídia) para reduzir o tempo de carregamento de ~15s para ~3-5s.
          2. Navega até a URL aguardando "networkidle" — estado em que não há
             requisições de rede por 500ms, indicando que o JS terminou de rodar.
          3. Faz scroll até a metade da página para disparar lazy loading de
             conteúdo que só carrega quando entra no viewport.
          4. Aguarda 1.5s adicional para renderizações assíncronas tardias.
          5. Captura o HTML final já com o DOM completo.

        Playwright é importado dentro do método (import tardio) para que o
        extractor funcione normalmente mesmo sem o Playwright instalado,
        só falhando ao tentar usar essa estratégia específica.
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            raise ExtractionError(
                "Playwright não instalado. Execute:\n"
                "  pip install playwright\n"
                "  python -m playwright install chromium"
            )

        with sync_playwright() as pw:
            # inicia Chromium em modo headless (sem interface gráfica)
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=TextExtractor.HEADERS["User-Agent"],
                locale="pt-BR",           # garante conteúdo em português
                java_script_enabled=True, # obrigatório para SPAs
            )
            page = context.new_page()

            # bloqueia recursos que não contribuem com texto
            # reduz significativamente o tempo de carregamento
            page.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3,avi}",
                lambda route: route.abort(),
            )

            try:
                # "networkidle": aguarda rede estabilizar antes de capturar o HTML
                # cobre a maioria dos SPAs que carregam dados via fetch/XHR
                page.goto(url, wait_until="networkidle", timeout=30_000)

                # scroll para metade da página — ativa lazy loading de artigos
                # que só carregam parágrafos quando o usuário chega até eles
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")

                # espera extra para renderizações assíncronas tardias
                page.wait_for_timeout(1500)

                # captura o HTML com o DOM completamente renderizado
                html = page.content()

            except PWTimeout:
                raise ExtractionError(f"Playwright timeout ao renderizar: {url}")
            finally:
                # sempre fecha o browser — evita processos zumbis
                browser.close()

        return html

    # =========================================================================
    # Parsing e extração de conteúdo
    # Métodos que transformam o HTML bruto em dados estruturados.
    # =========================================================================

    @staticmethod
    def _parse_html(html: str) -> BeautifulSoup:
        """
        Converte o HTML bruto em árvore BeautifulSoup e remove elementos
        de ruído definidos em TAGS_TO_REMOVE e SELECTORS_TO_REMOVE.

        A remoção acontece diretamente na árvore (decompose) antes de
        qualquer extração de texto, garantindo que scripts, estilos e
        elementos de UI não contaminem o conteúdo extraído.
        """
        soup = BeautifulSoup(html, "html.parser")

        # remove tags estruturais irrelevantes (script, style, nav etc.)
        for tag in soup(TextExtractor.TAGS_TO_REMOVE):
            tag.decompose()

        # remove elementos de ruído por seletor CSS (anúncios, popups etc.)
        for selector in TextExtractor.SELECTORS_TO_REMOVE:
            for el in soup.select(selector):
                el.decompose()

        return soup

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """
        Extrai o título da notícia seguindo uma hierarquia de confiabilidade:

          1. <h1 itemprop="headline"> — mais semântico e específico
          2. <h1> genérico            — presente na maioria dos portais
          3. <meta property="og:title"> — Open Graph, confiável em portais modernos
          4. <title>                  — último recurso; costuma conter sufixos
                                        como "| G1" ou "- Folha" que são removidos

        O <title> frequentemente vem no formato "Manchete | Nome do Portal —
        Cidade", então separamos pelo primeiro "|" ou " - " e ficamos só
        com a parte da manchete.
        """
        # tenta h1 com marcação semântica primeiro
        h1 = soup.find("h1", attrs={"itemprop": "headline"}) or soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Open Graph title — usado por portais que implementam og: tags
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # <title> como último recurso — remove sufixo do portal
        if soup.title and soup.title.string:
            raw = soup.title.string.strip()
            return raw.split("|")[0].split(" - ")[0].strip()

        return ""

    @staticmethod
    def _extract_meta_description(soup: BeautifulSoup) -> str:
        """
        Extrai a descrição/resumo da notícia a partir de metadados.
        Útil para o claim_detector.py identificar a afirmação principal
        sem precisar processar o artigo completo.

        Tenta na ordem:
          1. <meta name="description"> — padrão HTML
          2. <meta property="og:description"> — Open Graph
        """
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()

        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og["content"].strip()

        return ""

    @staticmethod
    def _extract_blocks(soup: BeautifulSoup) -> list[str]:
        """
        Extrai os blocos de texto do artigo em três camadas progressivas.

        O sistema de camadas garante cobertura máxima sem sacrificar precisão:
        tenta sempre o método mais confiável primeiro e só recorre ao mais
        genérico se o anterior não trouxer conteúdo suficiente (< 300 chars).

        Camada 1 — Roots semânticos
            Busca pelos contêineres de artigo mais confiáveis:
            itemprop="articleBody" > <article> > <main>
            Funciona para portais que seguem padrões semânticos HTML5.

        Camada 2 — Seletores específicos de portal
            Ativada quando os roots semânticos existem mas estão vazios
            ou com conteúdo insuficiente. Caso típico: G1 usa <article
            itemprop="articleBody"> mas envolve o texto em
            <div class="wall protected-content"> que é removido no _parse_html,
            deixando o article vazio. Os seletores específicos do portal
            (p.content-text__container) alcançam os parágrafos diretamente.

        Camada 3 — Fallback global
            Último recurso: todos os <p> da página. Produz mais ruído
            (menus, rodapés que escaparam da limpeza) mas garante que
            o pipeline nunca retorne completamente vazio em páginas válidas.

        Após coletar os candidatos, aplica dois filtros:
          - Descarta parágrafos dentro de elementos de paywall/wall
          - Descarta textos com menos de 40 caracteres (menus, labels, datas)
          - Deduplica via set para remover parágrafos repetidos
            (comum quando <article> e <main> compartilham o mesmo conteúdo)
        """
        seen: set[str] = set()
        blocks: list[str] = []

        # ── camada 1: roots semânticos ───────────────────────────────────────
        # itemprop="articleBody" é o sinal mais confiável de conteúdo principal
        # seguido de <article> e <main> como alternativas semânticas
        semantic_roots = (
            soup.find_all(attrs={"itemprop": "articleBody"})
            or soup.find_all("article")
            or soup.find_all("main")
        )

        # coleta todas as tags de texto dentro dos roots semânticos
        candidate_tags = []
        for root in semantic_roots:
            candidate_tags.extend(
                root.find_all(["p", "h2", "h3", "h4", "blockquote", "li"])
            )

        # ── camada 2: seletores específicos de portal ────────────────────────
        # verifica se os roots semânticos trouxeram conteúdo útil
        total_chars = sum(len(t.get_text(strip=True)) for t in candidate_tags)
        if total_chars < 300:
            # tenta cada seletor de portal em ordem de especificidade
            # para no primeiro que retornar conteúdo suficiente
            for selector in TextExtractor.PORTAL_PARAGRAPH_SELECTORS:
                found = soup.select(selector)
                found_chars = sum(len(t.get_text(strip=True)) for t in found)
                if found and found_chars >= 300:
                    candidate_tags = found
                    break

        # ── camada 3: fallback global ────────────────────────────────────────
        # só chega aqui se as camadas anteriores não encontraram nada
        if not candidate_tags:
            candidate_tags = soup.find_all("p")

        # ── filtragem e deduplicação ─────────────────────────────────────────
        for tag in candidate_tags:
            # descarta parágrafos dentro de elementos de paywall
            # (conteúdo bloqueado que passou pela limpeza de SELECTORS_TO_REMOVE)
            if tag.find_parent(class_=lambda c: c and (
                "wall" in c or "paywall" in c or "protected" in c
            )):
                continue

            # extrai texto limpo da tag, unindo textos de tags filhas com espaço
            text = tag.get_text(separator=" ", strip=True)

            # descarta textos muito curtos (datas, labels, breadcrumbs)
            # e duplicatas já vistas
            if len(text) < 40 or text in seen:
                continue

            seen.add(text)
            blocks.append(text)

        return blocks

    # =========================================================================
    # Ponto de entrada público
    # Orquestra todas as estratégias e retorna o resultado padronizado
    # consumido pelo pipeline.py.
    # =========================================================================

    @classmethod
    def extract(cls, url: str) -> dict:
        """
        Extrai o conteúdo textual de uma URL e retorna um dicionário
        padronizado para consumo pelo pipeline.py.

        Fluxo de execução:
          1. Valida a URL e verifica domínios não suportados
          2. Tenta requisição estática (requests)
          3. Se estática falhou → Playwright direto
          4. Se estática retornou SPA vazia → Playwright
          5. Detecta paywall e registra como warning
          6. Extrai título, descrição e blocos de texto
          7. Falha apenas se não houver título NEM blocos

        Retorno:
            url              — URL original
            title            — título do artigo
            description      — resumo/lead (meta description)
            content          — texto completo (blocos unidos por \n\n)
            content_blocks   — lista de blocos individuais (para segmentation.py)
            block_count      — número de blocos extraídos
            char_count       — total de caracteres extraídos
            render_method    — "static" | "playwright"
            paywall_detected — True se sinais de paywall foram encontrados
            warnings         — lista de avisos não-fatais ocorridos
        """
        cls._validate_url(url)

        if cls._is_unsupported_domain(url):
            raise ExtractionError(
                f"Domínio não suportado: {urlparse(url).hostname}. "
                "Redes sociais exigem APIs oficiais ou acesso autenticado."
            )

        warnings: list[str] = []
        render_method = "static"

        # ── tentativa 1: requisição estática ────────────────────────────────
        try:
            response = cls._fetch_static(url)
            # usa o encoding detectado automaticamente pelo chardet
            # evita lixo de caracteres em sites com charset mal declarado
            response.encoding = response.apparent_encoding
            html = response.text
        except ExtractionError as e:
            # estática falhou completamente (timeout, conexão, bloqueio definitivo)
            # registra o motivo e escala para Playwright
            warnings.append(f"Estratégia estática falhou ({e}), usando Playwright.")
            html = cls._fetch_with_playwright(url)
            render_method = "playwright"

        # parse e extração iniciais
        soup = cls._parse_html(html)
        blocks = cls._extract_blocks(soup)

        # ── tentativa 2: Playwright se SPA detectado ─────────────────────────
        # a estratégia estática pode ter "funcionado" (sem exceção) mas
        # retornado um HTML de SPA vazio — detectamos isso aqui
        if render_method == "static" and cls._needs_js_render(html, blocks):
            warnings.append("SPA detectado — usando Playwright para renderizar JS.")
            html = cls._fetch_with_playwright(url)
            render_method = "playwright"
            soup = cls._parse_html(html)
            blocks = cls._extract_blocks(soup)

        # ── detecção de paywall ──────────────────────────────────────────────
        # registra como warning, não erro — análise continua com o trecho livre
        paywall_detected = cls._detect_paywall(soup, html)
        if paywall_detected:
            warnings.append(
                "Paywall detectado: conteúdo pode estar incompleto. "
                "A análise será feita sobre o trecho disponível."
            )

        title = cls._extract_title(soup)
        description = cls._extract_meta_description(soup)

        # falha apenas se não tiver absolutamente nada — nem título nem texto
        # um título sem blocos ainda permite análise parcial pelo pipeline
        if not title and not blocks:
            raise ExtractionError(
                f"Nenhum conteúdo extraível encontrado em: {url}. "
                "A página pode exigir autenticação ou estar inacessível."
            )

        return {
            "url":              url,
            "title":            title,
            "description":      description,
            "content":          "\n\n".join(blocks),  # texto corrido para exibição
            "content_blocks":   blocks,                # blocos para segmentation.py
            "block_count":      len(blocks),
            "char_count":       sum(len(b) for b in blocks),
            "render_method":    render_method,         # "static" | "playwright"
            "paywall_detected": paywall_detected,
            "warnings":         warnings,
        }