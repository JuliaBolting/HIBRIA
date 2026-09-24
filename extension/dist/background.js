const API_URL = "https://hibria-tcc.duckdns.org";
const ANALYSIS_STATE_KEY = "hibria-analysis-state-v2";


async function updateStoredState(jobId, changes) {
  const values = await chrome.storage.local.get(ANALYSIS_STATE_KEY);
  const current = values[ANALYSIS_STATE_KEY];
  if (!current || current.jobId !== jobId) return;

  await chrome.storage.local.set({
    [ANALYSIS_STATE_KEY]: {
      ...current,
      ...changes,
    },
  });
}


async function startAnalysisJob(jobId, payload) {
  try {
    const response = await fetch(`${API_URL}/analyze/jobs/${jobId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data = null;
    try {
      data = await response.json();
    } catch {
      // A mensagem amigável abaixo cobre respostas sem JSON.
    }

    if (!response.ok) {
      throw new Error(
        data?.detail || data?.error || `Erro HTTP ${response.status}`
      );
    }

    await updateStoredState(jobId, {
      requestAccepted: true,
      startError: "",
      currentStep: Number(data?.current_step || 0),
    });

    return { success: true };
  } catch (error) {
    await updateStoredState(jobId, {
      requestAccepted: false,
      startError:
        error?.message || "Não foi possível iniciar a análise.",
    });
    return {
      success: false,
      error: error?.message || "Não foi possível iniciar a análise.",
    };
  }
}


async function cancelAnalysisJob(jobId) {
  try {
    await fetch(`${API_URL}/analyze/jobs/${jobId}`, {
      method: "DELETE",
    });
    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error?.message || "Não foi possível cancelar a análise.",
    };
  }
}


chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "START_ANALYSIS_JOB") {
    startAnalysisJob(message.jobId, message.payload).then(sendResponse);
    return true;
  }

  if (message?.type === "CANCEL_ANALYSIS_JOB") {
    cancelAnalysisJob(message.jobId).then(sendResponse);
    return true;
  }

  return false;
});
