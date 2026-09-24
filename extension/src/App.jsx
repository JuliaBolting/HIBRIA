import { useEffect, useState } from "react";
import { captureCurrentPage } from "./services/pageCapture";
import {
  cancelAnalysisJob,
  getAnalysisJob,
  startAnalysisJob,
  submitAnalysisFeedback,
} from "./services/api";
import {
  clearAnalysisState,
  createJobId,
  getOrCreateEvaluatorId,
  loadAnalysisState,
  loadSavedFeedback,
  saveAnalysisState,
  saveFeedback,
} from "./services/analysisState";
import "./App.css";


const STEPS = [
  {
    id: "capture",
    label: "Extraindo conteúdo",
    icon: "/file-text.png",
    color: "#8b44ef",
  },
  {
    id: "analysis",
    label: "Analisando linguagem",
    icon: "/languages.png",
    color: "#06bad1",
  },
  {
    id: "evidence",
    label: "Cruzando fontes externas",
    icon: "/globe.png",
    color: "#8b44ef",
  },
  {
    id: "explanation",
    label: "Gerando relatório",
    icon: "/chart-column.png",
    color: "#f59e0b",
  },
];


const WAITING_MESSAGES = [
  "Pegue uma xícara de café enquanto espera!",
  "Um cafezinho combina com uma boa investigação.",
  "Se preferir chá, a HÍBRIA não julga.",
  "A notícia está passando por uma investigação digital.",
  "Detetives digitais também precisam de alguns segundos.",
  "Sherlock Holmes aprovaria uma boa checagem de fontes.",
  "A lupa está trabalhando.",
  "Um título chamativo não conta a história inteira.",
  "Nem tudo que viraliza é verdade.",
  "Compartilhar é rápido. Verificar exige um pouco mais.",
  "Uma manchete merece mais que uma olhadinha.",
  "Já conferiu a data da notícia?",
  "Uma informação antiga pode parecer novidade.",
  "Antes de compartilhar, vale conferir a fonte.",
  "Uma imagem fora de contexto pode mudar uma história.",
  "Desconfie de títulos que provocam pressa.",
  "Uma boa pergunta pode valer mais que um clique.",
  "Comparar fontes ajuda a enxergar outros detalhes.",
  "Uma frase isolada pode perder seu contexto.",
  "A mesma notícia pode aparecer em lugares diferentes.",
  "Nem todo site tem o mesmo cuidado editorial.",
  "O contexto também faz parte da notícia.",
  "Uma notícia complexa pode exigir mais evidências.",
  "A dúvida também pode ser um bom começo.",
  "Pensamento crítico: ativado.",
  "Questionar é saudável. Investigar também.",
  "A curiosidade é uma ótima companheira da checagem.",
  "As fontes estão entrando na conversa.",
  "Estamos procurando pistas em diferentes fontes.",
  "Conferir informações leva mais tempo que ler um título.",
  "Cada evidência precisa ser interpretada no contexto.",
  "Um detalhe pequeno pode fazer diferença.",
  "Estamos juntando as peças desse quebra-cabeça.",
  "Às vezes, a resposta é: faltam evidências.",
  "Nem toda informação tem uma resposta simples.",
  "Um resultado responsável também reconhece limites.",
  "A tecnologia ajuda; seu senso crítico continua essencial.",
  "A HÍBRIA é uma ferramenta de apoio, não um oráculo.",
  "Você decide o que fazer com o resultado.",
  "Sempre vale consultar as fontes indicadas.",
  "Enquanto isso, pode continuar navegando à vontade.",
  "Fechar esta janela não interrompe a análise.",
  "Quando voltar, o andamento estará por aqui.",
  "A página atual só muda com uma nova análise.",
  "O botão de parar fica disponível se precisar.",
  "A análise continua mesmo com a janelinha fechada.",
  "Você pode aproveitar para esticar as pernas.",
  "Sua aba está liberada para outras aventuras.",
  "Pode voltar daqui a pouco; estamos trabalhando.",
  "Se a internet fosse simples, não haveria tanto para checar.",
  "Boatos não precisam de convite para se espalhar.",
  "Uma pesquisa cuidadosa pode mudar a perspectiva.",
  "A leitura crítica nunca sai de moda.",
  "HÍBRIA: tecnologia com uma dose de curiosidade.",
  "Sim, esta extensão faz parte de um TCC!",
  "Por trás da HÍBRIA, há muita pesquisa e desenvolvimento.",
  "Uma pequena extensão, um grande projeto de conclusão.",
  "Código, café e perguntas: a rotina de um TCC.",
  "Cada teste ajuda a aprimorar esta pesquisa.",
  "Você está conhecendo a HÍBRIA em funcionamento.",
  "Ciência também começa com boas perguntas.",
  "A ideia é aproximar tecnologia e leitura crítica.",
  "Seu olhar humano faz parte dessa experiência.",
  "O resultado chega com informações para você explorar.",
  "Que tal conferir as evidências ao final?",
  "Leia a explicação e confira os detalhes.",
  "A classificação é um ponto de partida, não o fim.",
  "Uma análise é mais útil quando entendemos seus motivos.",
  "Depois, conte o que achou da experiência.",
  "Sua avaliação ajuda a melhorar a HÍBRIA.",
  "Avaliar é opcional, mas seu retorno é valioso.",
  "Sua avaliação é anônima e ajuda nossa pesquisa.",
  "Um feedback curto pode render boas melhorias.",
  "Gostou da experiência? Conte para a gente!",
  "Encontrou algo estranho? Sua opinião importa.",
  "A pesquisa melhora quando escuta quem usa.",
  "Cada comentário pode inspirar uma nova ideia.",
  "A HÍBRIA está aprendendo com os testes e avaliações.",
  "Sua experiência pode contribuir com este TCC.",
  "Um pouquinho de paciência, uma dose de investigação.",
  "A pressa pode esperar; a evidência merece atenção.",
  "Notícias têm nuances. Vamos olhar com cuidado.",
  "Por aqui, a investigação segue em andamento.",
  "A checagem está trabalhando nos bastidores.",
  "Uma boa pausa para beber água.",
  "Respire, alongue os ombros e siga navegando.",
  "Uma página por vez; uma evidência de cada vez.",
  "Você trouxe a notícia. A HÍBRIA busca o contexto.",
  "Uma resposta explicada vale mais que um palpite.",
  "Pronto para olhar a notícia com outros olhos?",
];


const POLL_INTERVAL_MS = 1_800;


function App() {
  const [screen, setScreen] = useState("restoring");
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState("");
  const [requestPayload, setRequestPayload] = useState(null);
  const [messageIndex, setMessageIndex] = useState(0);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("hibria-theme") || "light"
  );


  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("hibria-theme", theme);
  }, [theme]);


  useEffect(() => {
    let active = true;

    async function restore() {
      const saved = await loadAnalysisState();
      if (!active) return;

      if (saved?.screen === "loading" && saved?.jobId) {
        setJobId(saved.jobId);
        setRequestPayload(saved.requestPayload || null);
        setCurrentStep(clampStep(saved.currentStep));
        setScreen("loading");
        return;
      }

      if (saved?.screen === "result" && saved?.result) {
        setJobId(saved.jobId || "");
        setResult(saved.result);
        setCurrentStep(3);
        setScreen("result");
        return;
      }

      setError(saved?.error || "");
      setScreen("home");
    }

    restore();
    return () => {
      active = false;
    };
  }, []);


  useEffect(() => {
    if (screen !== "loading") return undefined;

    const interval = window.setInterval(() => {
      setMessageIndex(
        (current) => (current + 1) % WAITING_MESSAGES.length
      );
    }, 4_200);

    return () => window.clearInterval(interval);
  }, [screen, jobId]);


  useEffect(() => {
    if (screen !== "loading" || !jobId) return undefined;

    let active = true;
    let timer = null;
    let consecutiveErrors = 0;
    let lastPersistedStep = null;

    function schedule(delay = POLL_INTERVAL_MS) {
      if (active) timer = window.setTimeout(poll, delay);
    }

    async function persistLoading(step) {
      if (step === lastPersistedStep) return;
      lastPersistedStep = step;
      await saveAnalysisState({
        screen: "loading",
        jobId,
        currentStep: step,
        requestPayload,
        startedAt: Date.now(),
      });
    }

    async function finishWithError(message) {
      if (!active) return;
      const friendly =
        message || "Não foi possível concluir a análise da página.";
      setError(friendly);
      setResult(null);
      setJobId("");
      setRequestPayload(null);
      setScreen("home");
      await saveAnalysisState({ screen: "home", error: friendly });
    }

    async function poll() {
      try {
        const job = await getAnalysisJob(jobId);
        if (!active) return;
        consecutiveErrors = 0;

        const step = clampStep(job.current_step);
        setCurrentStep(step);

        if (job.status === "completed") {
          const data = job.data;
          if (getScore(data) === null || !getResultLabel(data)) {
            await finishWithError(
              "O servidor devolveu uma análise incompleta. Tente novamente."
            );
            return;
          }

          const compact = compactResult(data);
          setResult(compact);
          setCurrentStep(3);
          setRequestPayload(null);
          setScreen("result");
          await saveAnalysisState({
            screen: "result",
            jobId,
            currentStep: 3,
            result: compact,
          });
          return;
        }

        if (job.status === "failed") {
          await finishWithError(job.error);
          return;
        }

        if (job.status === "cancelled") {
          await clearAnalysisState();
          if (!active) return;
          setScreen("home");
          setJobId("");
          setRequestPayload(null);
          return;
        }

        await persistLoading(step);
        schedule();
      } catch (pollError) {
        if (!active) return;

        // Se o popup foi fechado antes da confirmação do PUT, repetir o mesmo
        // UUID é seguro: o endpoint de criação é idempotente.
        if (pollError?.status === 404 && requestPayload) {
          try {
            await startAnalysisJob({ jobId, payload: requestPayload });
            consecutiveErrors = 0;
            schedule(700);
            return;
          } catch {
            // A tolerância abaixo evita falhar por uma oscilação curta de rede.
          }
        }

        consecutiveErrors += 1;
        if (consecutiveErrors >= 8) {
          await finishWithError(
            pollError?.message ||
              "Não foi possível consultar o andamento da análise."
          );
          return;
        }
        schedule(2_500);
      }
    }

    // Dá tempo para o service worker registrar o trabalho antes da primeira
    // consulta e evita repetir o PUT de criação sem necessidade.
    schedule(700);
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [screen, jobId, requestPayload]);


  function toggleTheme() {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }


  async function handleAnalyze() {
    setError("");
    setResult(null);
    setCurrentStep(0);
    setMessageIndex(0);

    try {
      const page = await captureCurrentPage();
      const newJobId = createJobId();
      const payload = {
        url: page.url,
        title: page.title,
        content: page.content,
      };
      const state = {
        screen: "loading",
        jobId: newJobId,
        currentStep: 0,
        requestPayload: payload,
        requestAccepted: false,
        startedAt: Date.now(),
      };

      // O estado é salvo antes da chamada de rede. Assim o mesmo trabalho pode
      // ser recuperado mesmo se a janela fechar imediatamente depois.
      await saveAnalysisState(state);
      setJobId(newJobId);
      setRequestPayload(payload);
      setScreen("loading");

      if (globalThis.chrome?.runtime?.sendMessage) {
        chrome.runtime
          .sendMessage({
            type: "START_ANALYSIS_JOB",
            jobId: newJobId,
            payload,
          })
          .catch(() => startAnalysisJob({ jobId: newJobId, payload }));
      } else {
        startAnalysisJob({ jobId: newJobId, payload }).catch(() => {});
      }
    } catch (captureError) {
      const message =
        captureError?.message ||
        "Não foi possível capturar o conteúdo da página.";
      setError(message);
      setScreen("home");
      await saveAnalysisState({ screen: "home", error: message });
    }
  }


  async function handleNewAnalysis() {
    const activeJobId = jobId;
    await clearAnalysisState();
    setScreen("home");
    setCurrentStep(0);
    setResult(null);
    setError("");
    setJobId("");
    setRequestPayload(null);

    if (activeJobId && screen === "loading") {
      if (globalThis.chrome?.runtime?.sendMessage) {
        chrome.runtime
          .sendMessage({
            type: "CANCEL_ANALYSIS_JOB",
            jobId: activeJobId,
          })
          .catch(() => cancelAnalysisJob(activeJobId));
      } else {
        cancelAnalysisJob(activeJobId).catch(() => {});
      }
    }
  }


  return (
    <div
      className={`app-shell view-${screen}`}
      style={{
        "--score": getScore(result) ?? 0,
        "--result-color": getResultColor(result),
      }}
    >
      <div className="header-controls">
        <button
          className="theme-button"
          type="button"
          onClick={toggleTheme}
          aria-label={
            theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"
          }
          title={
            theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"
          }
        >
          <img
            src={theme === "dark" ? "/sun.png" : "/moon.png"}
            alt=""
            className="theme-icon"
          />
        </button>

        <button
          className="close-button"
          type="button"
          onClick={() => window.close()}
          aria-label={
            screen === "loading"
              ? "Fechar a janela; a análise continuará"
              : "Fechar"
          }
          title="Fechar"
        >
          <img src="/fechar.png" alt="" />
        </button>
      </div>

      {screen === "restoring" && (
        <main className="screen screen-restoring" aria-live="polite">
          <img src="/logo-hibria.png" alt="HÍBRIA" className="logo-compact" />
          <p>Recuperando a última análise…</p>
        </main>
      )}

      {screen === "home" && (
        <main className="screen screen-home">
          <div className="brand">
            <img src="/logo-hibria.png" alt="HÍBRIA" className="logo" />
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
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
        </main>
      )}

      {screen === "loading" && (
        <main className="screen screen-loading">
          <img src="/logo-hibria.png" alt="HÍBRIA" className="logo-compact" />

          <div className="loading-content">
            <div className="thinking-dots" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </div>

            <h1>ANALISANDO...</h1>
            <p className="loading-status" aria-live="polite">
              {STEPS[currentStep]?.label}
            </p>
            <p key={messageIndex} className="waiting-message">
              {WAITING_MESSAGES[messageIndex]}
            </p>
          </div>

          <ol className="steps" aria-label="Etapas da análise">
            {STEPS.map((step, index) => {
              const isActive = index === currentStep;
              const isDone = index < currentStep;
              return (
                <li
                  key={step.id}
                  className={isActive ? "active" : isDone ? "done" : ""}
                  style={{ "--step-color": step.color }}
                  aria-current={isActive ? "step" : undefined}
                >
                  <span className="step-icon">
                    <img
                      src={isDone ? "/circle-check-big.png" : step.icon}
                      alt=""
                    />
                  </span>
                  <span className="step-label">{step.label}</span>
                </li>
              );
            })}
          </ol>

          <button
            className="stop-button"
            type="button"
            onClick={handleNewAnalysis}
          >
            <img src="/circle-stop.png" alt="" className="stop-icon" />
            <span>Parar Análise</span>
          </button>
        </main>
      )}

      {screen === "result" && (
        <main className="screen screen-result">
          <img src="/logo-hibria.png" alt="HÍBRIA" className="logo-compact" />
          <ScoreRing result={result} />

          <div className="verdict">
            <img src={getVerdictIcon(result)} alt="" />
            <span>{getVerdict(result)}</span>
          </div>

          <section className="result-card">
            <h2>
              <img src={getEvidenceIcon(result)} alt="" />
              <span>Evidências</span>
            </h2>
            <p>{getExplanation(result)}</p>
          </section>

          <section className="result-card">
            <h2>
              <img src={getDetailsIcon(result)} alt="" />
              <span>Detalhes</span>
            </h2>
            <ul>
              {getFactors(result).map((factor, index) => (
                <li key={index}>{factor}</li>
              ))}
            </ul>
          </section>

          <FeedbackCard key={getAnalysisId(result)} result={result} />

          <button
            className="new-analysis-button"
            type="button"
            onClick={handleNewAnalysis}
          >
            <img src="/rotate-ccw.png" alt="" />
            <span>Nova Análise</span>
          </button>
        </main>
      )}
    </div>
  );
}


function FeedbackCard({ result }) {
  const analysisId = getAnalysisId(result);
  const [rating, setRating] = useState(null);
  const [hovered, setHovered] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [feedbackError, setFeedbackError] = useState("");

  useEffect(() => {
    let active = true;

    loadSavedFeedback(analysisId).then((saved) => {
      if (active && saved) setRating(Number(saved));
    });

    return () => {
      active = false;
    };
  }, [analysisId]);

  async function chooseRating(value) {
    if (!analysisId || rating || submitting) return;
    setSubmitting(true);
    setFeedbackError("");

    try {
      const evaluatorId = await getOrCreateEvaluatorId(analysisId);
      const response = await submitAnalysisFeedback({
        analysisId,
        evaluatorId,
        rating: value,
      });
      const savedRating = Number(response?.rating || value);
      await saveFeedback(analysisId, savedRating);
      setRating(savedRating);
    } catch (submitError) {
      setFeedbackError(
        submitError?.message || "Não foi possível enviar sua avaliação."
      );
    } finally {
      setSubmitting(false);
    }
  }

  const highlighted = hovered || rating || 0;

  return (
    <section className="feedback-card" aria-labelledby="feedback-title">
      <h2 id="feedback-title">Como você avalia este resultado?</h2>
      <p>Sua avaliação é anônima e opcional.</p>

      {analysisId ? (
        <div
          className="star-rating"
          role="radiogroup"
          aria-label="Avaliação do resultado de uma a cinco estrelas"
          onMouseLeave={() => setHovered(0)}
        >
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              key={value}
              type="button"
              className={value <= highlighted ? "selected" : ""}
              onMouseEnter={() => !rating && setHovered(value)}
              onFocus={() => !rating && setHovered(value)}
              onBlur={() => setHovered(0)}
              onClick={() => chooseRating(value)}
              disabled={Boolean(rating) || submitting}
              role="radio"
              aria-checked={rating === value}
              aria-label={`${value} ${value === 1 ? "estrela" : "estrelas"}`}
              title={`${value} ${value === 1 ? "estrela" : "estrelas"}`}
            >
              ★
            </button>
          ))}
        </div>
      ) : (
        <p className="feedback-unavailable">
          A avaliação ficará disponível quando a análise estiver salva.
        </p>
      )}

      <div className="feedback-status" aria-live="polite">
        {submitting && "Enviando avaliação…"}
        {rating && "Obrigado! Sua avaliação foi registrada."}
        {feedbackError && <span role="alert">{feedbackError}</span>}
      </div>
    </section>
  );
}


function ScoreRing({ result }) {
  const score = getScore(result);
  return (
    <div
      className="score-ring"
      style={{
        "--score": score ?? 0,
        "--result-color": getResultColor(result),
      }}
    >
      <div className="score-content">
        <strong>{score === null ? "—" : `${formatScore(score)}%`}</strong>
        <span>confiabilidade</span>
      </div>
    </div>
  );
}


function compactResult(result) {
  return {
    analysis: result?.analysis || null,
    explanation: result?.explanation || "",
    details: Array.isArray(result?.details) ? result.details : [],
    evidence: result?.evidence
      ? {
          score: result.evidence.score,
          coverage: result.evidence.coverage,
          claim_count: result.evidence.claim_count,
          evidence_count: result.evidence.evidence_count,
        }
      : null,
    metadata: {
      cache: result?.metadata?.cache || null,
      explanation_source: result?.metadata?.explanation_source || null,
    },
  };
}


function clampStep(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(STEPS.length - 1, Math.trunc(number)));
}


function getAnalysisId(result) {
  return String(result?.metadata?.cache?.analysis_id || "").trim();
}


function getScore(result) {
  const score =
    result?.score ??
    result?.final_score ??
    result?.score_final ??
    result?.overall_score ??
    result?.hybrid_score ??
    result?.analysis?.score;

  if (score === undefined || score === null || score === "") return null;
  const numericScore = Number(score);
  if (!Number.isFinite(numericScore)) return null;
  return Math.max(0, Math.min(100, numericScore));
}


function formatScore(score) {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}


function getResultLabel(result) {
  const rawLabel =
    result?.analysis?.label ?? result?.label ?? result?.label_final;
  return String(rawLabel || "").trim().toLowerCase();
}


function getVerdict(result) {
  const label = getResultLabel(result);
  if (label) return formatStatus(label);

  const score = getScore(result);
  if (score === null) return "Resultado indisponível";
  if (score >= 70) return "Confiável";
  if (score >= 40) return "Parcialmente confiável";
  return "Não confiável";
}


function getVerdictIcon(result) {
  const label = getResultLabel(result);
  if (label === "confiável") return "/circle-check-big.png";
  if (label === "não confiável") return "/circle-x.png";
  return "/triangle-alert.png";
}


function getResultColor(result) {
  const label = getResultLabel(result);
  if (label === "confiável") return "#16c784";
  if (label === "não confiável") return "#ef4444";
  return "#f5a300";
}


function getEvidenceIcon(result) {
  const label = getResultLabel(result);
  if (label === "confiável") return "/clipboard-list-green.png";
  if (label === "não confiável") return "/clipboard-list-red.png";
  return "/clipboard-list-ambar.png";
}


function getDetailsIcon(result) {
  const label = getResultLabel(result);
  if (label === "confiável") return "/list-checks-green.png";
  if (label === "não confiável") return "/list-checks-red.png";
  return "/list-checks-ambar.png";
}


function getExplanation(result) {
  const explanation =
    result?.explanation ??
    result?.explanation_text ??
    result?.ai_explanation ??
    result?.analysis?.explanation;

  if (typeof explanation === "string" && explanation.trim()) {
    return explanation.trim();
  }
  return "Não foram encontradas informações suficientes para gerar uma explicação detalhada.";
}


function getFactors(result) {
  const factors =
    result?.details ??
    result?.factors ??
    result?.main_factors ??
    result?.key_factors ??
    result?.analysis?.details ??
    result?.analysis?.factors;

  if (Array.isArray(factors) && factors.length > 0) {
    return factors
      .map((factor) => {
        if (typeof factor === "string") return factor;
        if (factor?.description) return factor.description;
        if (factor?.name) return factor.name;
        return JSON.stringify(factor);
      })
      .slice(0, 4);
  }

  return [
    "Parte das informações não foi confirmada.",
    "As fontes consultadas não foram suficientes.",
    "O conteúdo foi verificado apenas parcialmente.",
  ];
}


function formatStatus(value) {
  const text = String(value).replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}


export default App;
