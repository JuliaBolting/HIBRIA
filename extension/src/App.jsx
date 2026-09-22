import { useEffect, useRef, useState } from "react";
import { captureCurrentPage } from "./services/pageCapture";
import { analyzePage } from "./services/api";
import "./App.css";

/* =========================================================
   ETAPAS DA ANÁLISE
   ========================================================= */

const STEPS = [
  {
    id: "capture",
    label: "Extraindo conteúdo",
    icon: "/file-text.png",
    color: "#8b44ef",
  },
  {
    id: "analysis",
    label: "Processando no servidor",
    icon: "/languages.png",
    color: "#06bad1",
  },
  {
    id: "validation",
    label: "Validando o resultado",
    icon: "/globe.png",
    color: "#8b44ef",
  },
];

/* =========================================================
   APP
   ========================================================= */

function App() {
  const [screen, setScreen] = useState("home");
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const abortControllerRef = useRef(null);

  const [theme, setTheme] = useState(
    () => localStorage.getItem("hibria-theme") || "light"
  );

  /* =======================================================
     TEMA
     ======================================================= */

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("hibria-theme", theme);
  }, [theme]);

  useEffect(
    () => () => abortControllerRef.current?.abort(),
    []
  );

  function toggleTheme() {
    setTheme((current) =>
      current === "dark" ? "light" : "dark"
    );
  }

  /* =======================================================
     INICIAR ANÁLISE
     ======================================================= */

  async function handleAnalyze() {
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setError("");
    setResult(null);
    setCurrentStep(0);
    setScreen("loading");

    try {
      /* -----------------------------------------------
         1. CAPTURA DA PÁGINA
         ----------------------------------------------- */

      setCurrentStep(0);

      const page = await captureCurrentPage();

      /* -----------------------------------------------
         2. ENVIO PARA API
         ----------------------------------------------- */

      setCurrentStep(1);

      const analysisPromise = analyzePage({
        url: page.url,
        title: page.title,
        content: page.content,
        signal: controller.signal,
      });

      const data = await analysisPromise;

      setCurrentStep(2);

      if (getScore(data) === null || !getResultLabel(data)) {
        throw new Error(
          "O servidor devolveu uma análise incompleta. Tente novamente."
        );
      }

      /* -----------------------------------------------
         5. RESULTADO
         ----------------------------------------------- */

      setResult(data);
      setScreen("result");
    } catch (err) {
      if (err?.name === "AbortError") {
        return;
      }
      console.error("Erro durante a análise:", err);

      setError(
        err?.message ||
          "Não foi possível concluir a análise da página."
      );

      setScreen("home");
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }

  /* =======================================================
     NOVA ANÁLISE
     ======================================================= */

  function handleNewAnalysis() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setScreen("home");
    setCurrentStep(0);
    setResult(null);
    setError("");
  }

  /* =======================================================
     RENDER
     ======================================================= */

  return (
    <div
      className={`app-shell view-${screen}`}
      style={{
        "--score": getScore(result) ?? 0,
        "--result-color": getResultColor(result),
      }}
    >
      {/* ===================================================
          CONTROLES SUPERIORES
          =================================================== */}

      <div className="header-controls">
        <button
          className="theme-button"
          type="button"
          onClick={toggleTheme}
          aria-label={
            theme === "dark"
              ? "Ativar tema claro"
              : "Ativar tema escuro"
          }
          title={
            theme === "dark"
              ? "Ativar tema claro"
              : "Ativar tema escuro"
          }
        >
          <img
            src={
              theme === "dark"
                ? "/sun.png"
                : "/moon.png"
            }
            alt=""
            className="theme-icon"
          />
        </button>

        <button
          className="close-button"
          type="button"
          onClick={() => window.close()}
          aria-label="Fechar"
          title="Fechar"
        >
          <img src="/fechar.png" alt="" />
        </button>
      </div>

      {/* ===================================================
          TELA INICIAL
          =================================================== */}

      {screen === "home" && (
        <main className="screen screen-home">
          <div className="brand">
            <img
              src="/logo-hibria.png"
              alt="HÍBRIA"
              className="logo"
            />

            <h1>HÍBRIA</h1>

            <p>
              Análise inteligente de
              <br />
              confiabilidade de conteúdo
            </p>
          </div>

          <button
            className="primary-button"
            type="button"
            onClick={handleAnalyze}
          >
            Iniciar Análise
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </main>
      )}

      {/* ===================================================
          TELA DE LOADING
          =================================================== */}

      {screen === "loading" && (
        <main className="screen screen-loading">
          <img
            src="/logo-hibria.png"
            alt="HÍBRIA"
            className="logo-compact"
          />

          <div className="loading-content">
            <div className="thinking-dots">
              <i />
              <i />
              <i />
              <i />
            </div>

            <h1>ANALISANDO...</h1>

            <p className="loading-status">
              {STEPS[currentStep]?.label}
            </p>
          </div>

          <ul className="steps">
            {STEPS.map((step, index) => {
              const isActive = index === currentStep;
              const isDone = index < currentStep;

              return (
                <li
                  key={step.id}
                  className={
                    isActive
                      ? "active"
                      : isDone
                      ? "done"
                      : ""
                  }
                  style={{
                    "--step-color": step.color,
                  }}
                >
                  <span className="step-icon">
                    {isDone ? (
                      <img
                        src="/circle-check-big.png"
                        alt=""
                      />
                    ) : (
                      <img
                        src={step.icon}
                        alt=""
                      />
                    )}
                  </span>

                  <span className="step-label">
                    {step.label}
                  </span>
                </li>
              );
            })}
          </ul>

          <button
            className="stop-button"
            type="button"
            onClick={handleNewAnalysis}
          >
            <img
              src="/circle-stop.png"
              alt=""
              className="stop-icon"
            />

            <span>Parar Análise</span>
          </button>
        </main>
      )}

      {/* ===================================================
          TELA DE RESULTADO
          =================================================== */}

      {screen === "result" && (
        <main className="screen screen-result">
          <img
            src="/logo-hibria.png"
            alt="HÍBRIA"
            className="logo-compact"
          />

          <ScoreRing result={result} />

          <div className="verdict">
            <img
              src={getVerdictIcon(result)}
              alt=""
            />

            <span>
              {getVerdict(result)}
            </span>
          </div>

          {/* -----------------------------------------------
              EVIDÊNCIAS
              ----------------------------------------------- */}

          <section className="result-card">
            <h2>
              <img
                src={getEvidenceIcon(result)}
                alt=""
              />

              <span>Evidências</span>
            </h2>

            <p>
              {getExplanation(result)}
            </p>
          </section>

          {/* -----------------------------------------------
              DETALHES
              ----------------------------------------------- */}

          <section className="result-card">
            <h2>
              <img
                src={getDetailsIcon(result)}
                alt=""
              />

              <span>Detalhes</span>
            </h2>

            <ul>
              {getFactors(result).map(
                (factor, index) => (
                  <li key={index}>
                    {factor}
                  </li>
                )
              )}
            </ul>
          </section>

          {/* -----------------------------------------------
              NOVA ANÁLISE
              ----------------------------------------------- */}

          <button
            className="new-analysis-button"
            type="button"
            onClick={handleNewAnalysis}
          >
            <img
              src="/rotate-ccw.png"
              alt=""
            />

            <span>Nova Análise</span>
          </button>
        </main>
      )}
    </div>
  );
}

/* =========================================================
   SCORE
   ========================================================= */

function ScoreRing({ result }) {
  const score = getScore(result);
  const color = getResultColor(result);

  return (
    <div
      className="score-ring"
      style={{
        "--score": score ?? 0,
        "--result-color": color,
      }}
    >
      <div className="score-content">
        <strong>
          {score === null ? "—" : `${formatScore(score)}%`}
        </strong>

        <span>confiabilidade</span>
      </div>
    </div>
  );
}

/* =========================================================
   SCORE
   ========================================================= */

function getScore(result) {
  const score =
    result?.score ??
    result?.final_score ??
    result?.score_final ??
    result?.overall_score ??
    result?.hybrid_score ??
    result?.analysis?.score;

  if (score === undefined || score === null || score === "") {
    return null;
  }

  const numericScore = Number(score);

  if (!Number.isFinite(numericScore)) {
    return null;
  }

  return Math.max(
    0,
    Math.min(100, numericScore)
  );
}

function formatScore(score) {
  return Number.isInteger(score)
    ? String(score)
    : score.toFixed(1);
}

/* =========================================================
   VEREDITO
   ========================================================= */

function getResultLabel(result) {
  const rawLabel =
    result?.analysis?.label ??
    result?.label ??
    result?.label_final;

  return String(rawLabel || "")
    .trim()
    .toLowerCase();
}

function getVerdict(result) {
  const label = getResultLabel(result);

  if (label) {
    return formatStatus(label);
  }

  const score = getScore(result);

  if (score === null) return "Resultado indisponível";

  if (score >= 70) return "Confiável";
  if (score >= 40) return "Parcialmente confiável";
  return "Não confiável";
}

function getVerdictIcon(result) {
  const label = getResultLabel(result);

  if (label === "confiável") {
    return "/circle-check-big.png";
  }

  if (label === "não confiável") {
    return "/circle-x.png";
  }

  return "/triangle-alert.png";
}

/* =========================================================
   CORES
   ========================================================= */

function getResultColor(result) {
  const label = getResultLabel(result);

  if (label === "confiável") {
    return "#16c784";
  }

  if (label === "não confiável") {
    return "#ef4444";
  }

  return "#f5a300";
}

/* =========================================================
   ÍCONES DO RESULTADO
   ========================================================= */

function getEvidenceIcon(result) {
  const label = getResultLabel(result);

  if (label === "confiável") {
    return "/clipboard-list-green.png";
  }

  if (label === "não confiável") {
    return "/clipboard-list-red.png";
  }

  return "/clipboard-list-ambar.png";
}

function getDetailsIcon(result) {
  const label = getResultLabel(result);

  if (label === "confiável") {
    return "/list-checks-green.png";
  }

  if (label === "não confiável") {
    return "/list-checks-red.png";
  }

  return "/list-checks-ambar.png";
}

/* =========================================================
   EXPLICAÇÃO
   ========================================================= */

function getExplanation(result) {
  const explanation =
    result?.explanation ??
    result?.explanation_text ??
    result?.ai_explanation ??
    result?.analysis?.explanation;

  if (
    explanation &&
    typeof explanation === "string" &&
    explanation.trim()
  ) {
    return explanation.trim();
  }

  return "A análise do HÍBRIA não encontrou informações suficientes para gerar uma explicação detalhada.";
}

/* =========================================================
   FATORES
   ========================================================= */

function getFactors(result) {
  const factors =
    result?.details ??
    result?.factors ??
    result?.main_factors ??
    result?.key_factors ??
    result?.analysis?.details ??
    result?.analysis?.factors;

  if (
    Array.isArray(factors) &&
    factors.length > 0
  ) {
    return factors
      .map((factor) => {
        if (typeof factor === "string") {
          return factor;
        }

        if (factor?.description) {
          return factor.description;
        }

        if (factor?.name) {
          return factor.name;
        }

        return JSON.stringify(factor);
      })
      .slice(0, 4);
  }

  const evidenceScore =
    result?.evidence?.score ??
    result?.evidence_score ??
    result?.analysis?.evidence_score;

  const reputationStatus =
    result?.source?.reputation?.status ??
    result?.reputation?.status;

  const factorsFound = [];

  if (evidenceScore !== undefined) {
    factorsFound.push(
      `Score de evidências: ${evidenceScore}.`
    );
  }

  if (reputationStatus) {
    factorsFound.push(
      `Reputação da fonte: ${formatStatus(
        reputationStatus
      )}.`
    );
  }

  if (factorsFound.length > 0) {
    return factorsFound;
  }

  return [
    "Parte das informações não foi confirmada.",
    "Fontes externas insuficientes.",
    "Dados parcialmente verificáveis.",
  ];
}

/* =========================================================
   FORMATAÇÃO
   ========================================================= */

function formatStatus(value) {
  const text = String(value).replaceAll(
    "_",
    " "
  );

  return (
    text.charAt(0).toUpperCase() +
    text.slice(1)
  );
}

export default App;
