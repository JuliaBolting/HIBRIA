# responsável por extrair o conteúdo textual relevante de uma página web

import requests
from bs4 import BeautifulSoup


class TextExtractor:

    @staticmethod
    def extract(url: str):

        # realiza requisição HTTP da página
        response = requests.get(
            url,
            headers={
                # simula um navegador real para evitar bloqueios simples
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10  # tempo máximo de espera da requisição
        )

        # converte o HTML da página em uma estrutura navegável
        soup = BeautifulSoup(response.text, "html.parser")

        # remove elementos irrelevantes para análise textual
        for tag in soup([
            "img",
            "script",
            "style",
            "iframe",
            "ads",
            "footer",
            "nav",
            "header",
            "aside"
        ]):
            tag.decompose()

        # extrai o título da página
        title = soup.title.string.strip() if soup.title else ""

        # lista que armazenará os blocos de texto extraídos
        content = []

        # busca elementos de artigo e parágrafos
        for tag in soup.find_all(["article", "p"]):

            # extrai apenas o texto limpo da tag
            text = tag.get_text(separator=" ", strip=True)

            # ignora textos muito pequenos para reduzir ruído
            if len(text) > 30:
                content.append(text)

        # retorna os dados estruturados da página
        return {
            "url": url,
            "title": title,
            "content": "\n".join(content)
        }