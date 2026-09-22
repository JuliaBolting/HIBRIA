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

-- =============================================================================
-- Cache e histórico das análises da HÍBRIA
-- =============================================================================

CREATE TABLE IF NOT EXISTS analises (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url_original TEXT NOT NULL,
    url_normalizada TEXT NOT NULL,
    hash_url CHAR(64) NOT NULL,
    titulo TEXT,
    conteudo TEXT NOT NULL,
    hash_conteudo CHAR(64) NOT NULL,
    versao_pipeline VARCHAR(80) NOT NULL,
    score_final NUMERIC(6,2),
    classificacao_final VARCHAR(80),
    explicacao TEXT,
    tempo_processamento_segundos NUMERIC(12,3),
    resultado_json JSONB NOT NULL,
    data_analise TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ultima_consulta_em TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quantidade_consultas INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT uq_analises_cache
        UNIQUE (hash_url, hash_conteudo, versao_pipeline),
    CONSTRAINT ck_analises_hash_url
        CHECK (char_length(hash_url) = 64),
    CONSTRAINT ck_analises_hash_conteudo
        CHECK (char_length(hash_conteudo) = 64),
    CONSTRAINT ck_analises_score
        CHECK (score_final IS NULL OR score_final BETWEEN 0 AND 100),
    CONSTRAINT ck_analises_quantidade_consultas
        CHECK (quantidade_consultas >= 1)
);

CREATE INDEX IF NOT EXISTS idx_analises_url_normalizada
    ON analises (hash_url);

CREATE INDEX IF NOT EXISTS idx_analises_data
    ON analises (data_analise DESC);

CREATE INDEX IF NOT EXISTS idx_analises_classificacao
    ON analises (classificacao_final);

-- =============================================================================
-- Memória persistente do RAG
-- =============================================================================

CREATE TABLE IF NOT EXISTS claims_analise (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analise_id UUID NOT NULL REFERENCES analises(id) ON DELETE CASCADE,
    claim_id_pipeline VARCHAR(80) NOT NULL,
    ordem INTEGER NOT NULL,
    texto TEXT NOT NULL,
    texto_normalizado TEXT,
    hash_claim CHAR(64) NOT NULL,
    sujeito TEXT,
    confianca NUMERIC(6,5),
    entidades_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_criacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_claims_analise_pipeline UNIQUE (analise_id, claim_id_pipeline),
    CONSTRAINT ck_claims_analise_hash CHECK (char_length(hash_claim) = 64),
    CONSTRAINT ck_claims_analise_confianca CHECK (confianca IS NULL OR confianca BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_claims_analise_analise ON claims_analise (analise_id, ordem);
CREATE INDEX IF NOT EXISTS idx_claims_analise_hash ON claims_analise (hash_claim);

CREATE TABLE IF NOT EXISTS evidencias_rag (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_key CHAR(64) NOT NULL UNIQUE,
    url TEXT,
    url_normalizada TEXT,
    hash_url CHAR(64),
    titulo TEXT,
    trecho TEXT NOT NULL,
    hash_conteudo CHAR(64) NOT NULL,
    fonte TEXT,
    dominio TEXT,
    tipo_fonte VARCHAR(80),
    data_publicacao TEXT,
    fonte_confiavel BOOLEAN NOT NULL DEFAULT FALSE,
    aprovada_para_rag BOOLEAN NOT NULL DEFAULT FALSE,
    motivo_aprovacao TEXT,
    faiss_doc_id TEXT,
    faiss_indexada_em TIMESTAMPTZ,
    tentativas_indexacao INTEGER NOT NULL DEFAULT 0,
    ultimo_erro_indexacao TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    primeira_observacao_em TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ultima_observacao_em TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quantidade_usos INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT ck_evidencias_rag_key CHECK (char_length(evidence_key) = 64),
    CONSTRAINT ck_evidencias_rag_hash_conteudo CHECK (char_length(hash_conteudo) = 64),
    CONSTRAINT ck_evidencias_rag_hash_url CHECK (hash_url IS NULL OR char_length(hash_url) = 64),
    CONSTRAINT ck_evidencias_rag_quantidade_usos CHECK (quantidade_usos >= 1),
    CONSTRAINT ck_evidencias_rag_tentativas CHECK (tentativas_indexacao >= 0)
);

CREATE INDEX IF NOT EXISTS idx_evidencias_rag_dominio ON evidencias_rag (dominio);
CREATE INDEX IF NOT EXISTS idx_evidencias_rag_pendentes
    ON evidencias_rag (aprovada_para_rag, faiss_indexada_em)
    WHERE aprovada_para_rag = TRUE AND faiss_indexada_em IS NULL;

CREATE TABLE IF NOT EXISTS claim_evidencias (
    analise_id UUID NOT NULL REFERENCES analises(id) ON DELETE CASCADE,
    claim_id UUID NOT NULL REFERENCES claims_analise(id) ON DELETE CASCADE,
    evidencia_id UUID NOT NULL REFERENCES evidencias_rag(id) ON DELETE RESTRICT,
    evidence_id_pipeline VARCHAR(80),
    camada_recuperacao VARCHAR(80),
    ordem INTEGER NOT NULL,
    similaridade_retriever NUMERIC(6,5),
    similaridade_final NUMERIC(6,5),
    stance VARCHAR(30),
    stance_confianca NUMERIC(6,5),
    stance_motivo TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_criacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (analise_id, claim_id, evidencia_id),
    CONSTRAINT ck_claim_evidencias_sim_retriever CHECK (similaridade_retriever IS NULL OR similaridade_retriever BETWEEN 0 AND 1),
    CONSTRAINT ck_claim_evidencias_sim_final CHECK (similaridade_final IS NULL OR similaridade_final BETWEEN 0 AND 1),
    CONSTRAINT ck_claim_evidencias_stance_confianca CHECK (stance_confianca IS NULL OR stance_confianca BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_claim_evidencias_claim ON claim_evidencias (claim_id, ordem);
CREATE INDEX IF NOT EXISTS idx_claim_evidencias_evidencia ON claim_evidencias (evidencia_id);
