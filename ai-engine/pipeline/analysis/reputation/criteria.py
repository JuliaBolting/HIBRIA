from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReputationCriterion:
    key: str
    label: str
    weight: int
    query_templates: tuple[str, ...]
    positive_terms: tuple[str, ...]
    direct_paths: tuple[str, ...] = ()
    use_factcheck: bool = False


CRITERIA: tuple[ReputationCriterion, ...] = (
    ReputationCriterion(
        key="identificacao_institucional",
        label="Identificação institucional",
        weight=15,
        query_templates=(
            'site:{domain} ("quem somos" OR institucional OR expediente OR contato)',
        ),
        positive_terms=(
            "quem somos", "institucional", "expediente", "sobre nós", "sobre nos",
            "redação", "redacao", "contato", "empresa responsável", "empresa responsavel",
        ),
        direct_paths=("/quem-somos", "/sobre", "/sobre-nos", "/institucional", "/expediente", "/contato"),
    ),
    ReputationCriterion(
        key="autoria_responsabilidade_editorial",
        label="Autoria e responsabilidade editorial",
        weight=10,
        query_templates=(
            'site:{domain} (autor OR autoria OR editoria OR jornalistas OR expediente)',
        ),
        positive_terms=(
            "autor", "autoria", "jornalista", "repórter", "reporter", "editor",
            "editoria", "diretor de redação", "diretor de redacao", "equipe",
        ),
        direct_paths=("/expediente", "/equipe", "/redacao", "/redação"),
    ),
    ReputationCriterion(
        key="politica_correcoes_canal_erros",
        label="Política de correções ou canal para erros",
        weight=10,
        query_templates=(
            'site:{domain} (correções OR correcoes OR errata OR erros OR ombudsman OR "fale conosco")',
        ),
        positive_terms=(
            "correção", "correcao", "correções", "correcoes", "errata", "ombudsman",
            "fale conosco", "comunicar erro", "reportar erro", "política de correções",
        ),
        direct_paths=("/correcoes", "/correções", "/errata", "/ombudsman", "/fale-conosco", "/politica-de-correcoes"),
    ),
    ReputationCriterion(
        key="separacao_noticia_opiniao_publicidade",
        label="Separação entre notícia, opinião e publicidade",
        weight=10,
        query_templates=(
            'site:{domain} (opinião OR opiniao OR editorial OR publicidade OR patrocinado OR publieditorial)',
        ),
        positive_terms=(
            "opinião", "opiniao", "editorial", "colunista", "publicidade", "anuncie",
            "conteúdo patrocinado", "conteudo patrocinado", "publieditorial", "publicitário",
        ),
        direct_paths=("/opiniao", "/opinião", "/editorial", "/publicidade", "/anuncie", "/termos-de-uso"),
    ),
    ReputationCriterion(
        key="participacao_iniciativas_checagem",
        label="Participação em iniciativas de checagem",
        weight=15,
        query_templates=(
            '"{source_name}" (checagem OR "Projeto Comprova" OR verifica OR "fact-checking" OR IFCN)',
        ),
        positive_terms=(
            "checagem", "projeto comprova", "fact-checking", "fact checking", "ifcn",
            "verifica", "verificação de fatos", "verificacao de fatos",
        ),
        use_factcheck=True,
    ),
    ReputationCriterion(
        key="associacao_reconhecimento_jornalistico",
        label="Associação ou reconhecimento jornalístico",
        weight=10,
        query_templates=(
            '"{source_name}" (ANJ OR FENAJ OR associação OR premio OR prêmio OR jornalismo)',
        ),
        positive_terms=(
            "associação nacional de jornais", "associacao nacional de jornais", "anj",
            "federação nacional dos jornalistas", "federacao nacional dos jornalistas", "fenaj",
            "prêmio", "premio", "premiação", "premiacao", "reconhecimento jornalístico",
        ),
    ),
    ReputationCriterion(
        key="historico_publico_desinformacao",
        label="Histórico público de desinformação",
        weight=20,
        query_templates=(
            '"{source_name}" (desinformação OR "fake news" OR boato OR correção OR checagem)',
            '"{domain}" (desinformação OR "notícias falsas" OR "fake news")',
        ),
        positive_terms=(
            "desinformação", "desinformacao", "fake news", "notícias falsas", "noticias falsas",
            "boato", "enganoso", "falso", "correção", "correcao", "desmentido",
        ),
        use_factcheck=True,
    ),
    ReputationCriterion(
        key="adequacao_tecnica_sistema",
        label="Adequação técnica ao sistema",
        weight=10,
        query_templates=(),
        positive_terms=(),
        direct_paths=("/",),
    ),
)

CRITERIA_BY_KEY = {criterion.key: criterion for criterion in CRITERIA}
TOTAL_WEIGHT = sum(criterion.weight for criterion in CRITERIA)


def weights_output() -> dict[str, dict[str, int | str]]:
    return {
        criterion.key: {"label": criterion.label, "weight": criterion.weight}
        for criterion in CRITERIA
    }
