const ANALYSIS_STATE_KEY = "hibria-analysis-state-v2";
const EVALUATOR_IDS_KEY = "hibria-feedback-evaluator-ids-v1";
const FEEDBACKS_KEY = "hibria-feedbacks-v1";


function hasChromeStorage() {
  return Boolean(globalThis.chrome?.storage?.local);
}


async function readValue(key) {
  if (hasChromeStorage()) {
    const values = await chrome.storage.local.get(key);
    return values[key] ?? null;
  }

  try {
    return JSON.parse(localStorage.getItem(key) || "null");
  } catch {
    return null;
  }
}


async function writeValue(key, value) {
  if (hasChromeStorage()) {
    await chrome.storage.local.set({ [key]: value });
    return;
  }
  localStorage.setItem(key, JSON.stringify(value));
}


async function removeValue(key) {
  if (hasChromeStorage()) {
    await chrome.storage.local.remove(key);
    return;
  }
  localStorage.removeItem(key);
}


function createUuid() {
  if (globalThis.crypto?.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
    /[xy]/g,
    (character) => {
      const random = Math.floor(Math.random() * 16);
      const value = character === "x" ? random : (random & 0x3) | 0x8;
      return value.toString(16);
    }
  );
}


export function createJobId() {
  return createUuid();
}


export function loadAnalysisState() {
  return readValue(ANALYSIS_STATE_KEY);
}


export function saveAnalysisState(state) {
  return writeValue(ANALYSIS_STATE_KEY, state);
}


export function clearAnalysisState() {
  return removeValue(ANALYSIS_STATE_KEY);
}


export async function getOrCreateEvaluatorId(analysisId) {
  const evaluatorIds = (await readValue(EVALUATOR_IDS_KEY)) || {};
  if (evaluatorIds[analysisId]) return evaluatorIds[analysisId];
  const evaluatorId = createUuid();
  evaluatorIds[analysisId] = evaluatorId;
  await writeValue(EVALUATOR_IDS_KEY, evaluatorIds);
  return evaluatorId;
}


export async function loadSavedFeedback(analysisId) {
  if (!analysisId) return null;
  const feedbacks = (await readValue(FEEDBACKS_KEY)) || {};
  return feedbacks[analysisId] ?? null;
}


export async function saveFeedback(analysisId, rating) {
  const feedbacks = (await readValue(FEEDBACKS_KEY)) || {};
  feedbacks[analysisId] = rating;
  await writeValue(FEEDBACKS_KEY, feedbacks);
}
