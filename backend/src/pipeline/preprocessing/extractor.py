import requests
from bs4 import BeautifulSoup

class TextExtractor:

    @staticmethod
    def extract(url: str):

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )

        soup = BeautifulSoup(response.text, "html.parser")

        # remove elementos irrelevantes
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

        # título
        title = soup.title.string.strip() if soup.title else ""

        # captura conteúdo
        content = []

        for tag in soup.find_all(["article", "p"]):

            text = tag.get_text(separator=" ", strip=True)

            if len(text) > 30:
                content.append(text)

        return {
            "url": url,
            "title": title,
            "content": "\n".join(content)
        }