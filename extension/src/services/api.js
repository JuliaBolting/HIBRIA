export const API_URL =
  import.meta.env.VITE_HIBRIA_API_URL ||
  "https://hibria-tcc.duckdns.org";


async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Resposta inválida do servidor (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    const error = new Error(
      data?.error || data?.detail || `Erro HTTP ${response.status}`
    );
    error.status = response.status;
    throw error;
  }

  return data;
}


export async function analyzePage({ url, title, content, signal }) {
  const data = await apiRequest("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, title, content }),
    signal,
  });

  if (data?.success === false) {
    throw new Error(
      data.error || "A HÍBRIA não conseguiu analisar a página."
    );
  }

  return data?.data ?? data;
}


export function startAnalysisJob({ jobId, payload }) {
  return apiRequest(`/analyze/jobs/${jobId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}


export function getAnalysisJob(jobId) {
  return apiRequest(`/analyze/jobs/${jobId}`);
}


export function cancelAnalysisJob(jobId) {
  return apiRequest(`/analyze/jobs/${jobId}`, {
    method: "DELETE",
  });
}


export function submitAnalysisFeedback({
  analysisId,
  evaluatorId,
  rating,
}) {
  return apiRequest("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_id: analysisId,
      evaluator_id: evaluatorId,
      rating,
    }),
  });
}
