export async function captureCurrentPage() {
  const tabs = await chrome.tabs.query({
    active: true,
    currentWindow: true,
  });

  const tab = tabs[0];

  if (!tab?.id) {
    throw new Error("Não foi possível encontrar a aba atual.");
  }

  if (
    !tab.url ||
    tab.url.startsWith("chrome://") ||
    tab.url.startsWith("edge://") ||
    tab.url.startsWith("about:")
  ) {
    throw new Error(
      "A HÍBRIA não pode analisar esta página do navegador."
    );
  }

  const [result] = await chrome.scripting.executeScript({
    target: {
      tabId: tab.id,
    },

    func: () => {
      const title =
        document.querySelector("h1")?.innerText?.trim() ||
        document.title?.trim() ||
        "";

      const selectors = [
        "article",
        '[itemprop="articleBody"]',
        "main",
        '[role="main"]',
      ];

      let content = "";

      for (const selector of selectors) {
        const element = document.querySelector(selector);

        if (element) {
          const text = element.innerText?.trim() || "";

          if (text.length > content.length) {
            content = text;
          }
        }
      }

      if (!content) {
        content = document.body?.innerText?.trim() || "";
      }

      return {
        url: window.location.href,
        title,
        content,
      };
    },
  });

  if (!result?.result) {
    throw new Error(
      "Não foi possível capturar o conteúdo da página."
    );
  }

  const page = result.result;

  if (!page.content || page.content.trim().length < 100) {
    throw new Error(
      "Não foi encontrado conteúdo suficiente para análise."
    );
  }

  return page;
}