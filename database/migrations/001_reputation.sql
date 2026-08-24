-- =============================================================================
-- Reputação dinâmica de fontes da HÍBRIA
--
-- Não existe lista fixa de fontes confiáveis. Cada domínio é avaliado pelos
-- critérios ponderados do TCC e o resultado é persistido para reutilização.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS fontes_reputacao (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dominio_canonico TEXT NOT NULL UNIQUE,
    nome_fonte TEXT,
    status_avaliacao VARCHAR(40) NOT NULL,
    nota_total SMALLINT,
    score_reputacao NUMERIC(5,4) NOT NULL DEFAULT 0.5000,
    classificacao VARCHAR(80) NOT NULL,
    origem VARCHAR(50) NOT NULL,
    metodo_avaliacao VARCHAR(100) NOT NULL,
    precisa_revisao BOOLEAN NOT NULL DEFAULT TRUE,
    data_avaliacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_ultima_verificacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    proxima_reavaliacao TIMESTAMPTZ,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT ck_fontes_reputacao_nota
        CHECK (nota_total IS NULL OR nota_total BETWEEN 0 AND 100),
    CONSTRAINT ck_fontes_reputacao_score
        CHECK (score_reputacao BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_fontes_reputacao_status
    ON fontes_reputacao (status_avaliacao);

CREATE INDEX IF NOT EXISTS idx_fontes_reputacao_reavaliacao
    ON fontes_reputacao (proxima_reavaliacao);

CREATE TABLE IF NOT EXISTS aliases_fontes_reputacao (
    dominio_alias TEXT PRIMARY KEY,
    fonte_id UUID NOT NULL REFERENCES fontes_reputacao(id) ON DELETE CASCADE,
    tipo_alias VARCHAR(60) NOT NULL DEFAULT 'automatic_redirect_or_canonical',
    ativo BOOLEAN NOT NULL DEFAULT TRUE,
    data_criacao TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_aliases_fontes_fonte_id
    ON aliases_fontes_reputacao (fonte_id);

CREATE TABLE IF NOT EXISTS criterios_reputacao_fonte (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fonte_id UUID NOT NULL REFERENCES fontes_reputacao(id) ON DELETE CASCADE,
    criterio VARCHAR(100) NOT NULL,
    peso_maximo SMALLINT NOT NULL,
    pontos_obtidos SMALLINT,
    status VARCHAR(50) NOT NULL,
    justificativa TEXT,
    data_criacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (fonte_id, criterio),
    CONSTRAINT ck_criterio_peso CHECK (peso_maximo BETWEEN 0 AND 100),
    CONSTRAINT ck_criterio_pontos CHECK (
        pontos_obtidos IS NULL OR pontos_obtidos BETWEEN 0 AND peso_maximo
    )
);

CREATE TABLE IF NOT EXISTS evidencias_reputacao_fonte (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fonte_id UUID NOT NULL REFERENCES fontes_reputacao(id) ON DELETE CASCADE,
    criterio VARCHAR(100) NOT NULL,
    provedor VARCHAR(60) NOT NULL,
    tipo_evidencia VARCHAR(60) NOT NULL,
    titulo TEXT,
    url TEXT NOT NULL,
    trecho TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_coleta TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidencias_reputacao_fonte_id
    ON evidencias_reputacao_fonte (fonte_id);

CREATE TABLE IF NOT EXISTS execucoes_avaliacao_reputacao (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dominio_solicitado TEXT NOT NULL,
    dominio_canonico TEXT,
    gatilho VARCHAR(40) NOT NULL,
    status VARCHAR(40) NOT NULL,
    provedores_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    quantidade_consultas INTEGER NOT NULL DEFAULT 0,
    quantidade_evidencias INTEGER NOT NULL DEFAULT 0,
    mensagem_erro TEXT,
    iniciada_em TIMESTAMPTZ,
    finalizada_em TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
