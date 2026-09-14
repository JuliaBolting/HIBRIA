const API_URL = "http://127.0.0.1:8000";

export async function analyzePage({
  url,
  title,
  content,
  signal,
}) {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      url,
      title,
      content,
    }),
    signal,
  });

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Resposta inválida do servidor (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    throw new Error(
      data?.error ||
        data?.detail ||
        `Erro HTTP ${response.status}`
    );
  }

  if (data?.success === false) {
    throw new Error(
      data.error ||
        "A HÍBRIA não conseguiu analisar a página."
    );
  }

  /*
   * O backend pode retornar:
   *
   * { success: true, data: {...} }
   *
   * ou diretamente:
   *
   * {...}
   *
   * Aceitamos os dois formatos.
   */
  return data?.data ?? data;
}