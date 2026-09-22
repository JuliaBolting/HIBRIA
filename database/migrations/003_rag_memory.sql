-- =============================================================================
-- Memória persistente do RAG da HÍBRIA
--
-- PostgreSQL guarda as claims, as evidências e a relação observada em cada
-- análise. O FAISS continua guardando apenas os vetores e um identificador que
-- aponta para evidencias_rag.id.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
    CONSTRAINT uq_claims_analise_pipeline
        UNIQUE (analise_id, claim_id_pipeline),
    CONSTRAINT ck_claims_analise_hash
        CHECK (char_length(hash_claim) = 64),
    CONSTRAINT ck_claims_analise_confianca
        CHECK (confianca IS NULL OR confianca BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_claims_analise_analise
    ON claims_analise (analise_id, ordem);

CREATE INDEX IF NOT EXISTS idx_claims_analise_hash
    ON claims_analise (hash_claim);

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
    CONSTRAINT ck_evidencias_rag_key
        CHECK (char_length(evidence_key) = 64),
    CONSTRAINT ck_evidencias_rag_hash_conteudo
        CHECK (char_length(hash_conteudo) = 64),
    CONSTRAINT ck_evidencias_rag_hash_url
        CHECK (hash_url IS NULL OR char_length(hash_url) = 64),
    CONSTRAINT ck_evidencias_rag_quantidade_usos
        CHECK (quantidade_usos >= 1),
    CONSTRAINT ck_evidencias_rag_tentativas
        CHECK (tentativas_indexacao >= 0)
);

CREATE INDEX IF NOT EXISTS idx_evidencias_rag_dominio
    ON evidencias_rag (dominio);

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
    CONSTRAINT ck_claim_evidencias_sim_retriever
        CHECK (similaridade_retriever IS NULL OR similaridade_retriever BETWEEN 0 AND 1),
    CONSTRAINT ck_claim_evidencias_sim_final
        CHECK (similaridade_final IS NULL OR similaridade_final BETWEEN 0 AND 1),
    CONSTRAINT ck_claim_evidencias_stance_confianca
        CHECK (stance_confianca IS NULL OR stance_confianca BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_claim_evidencias_claim
    ON claim_evidencias (claim_id, ordem);

CREATE INDEX IF NOT EXISTS idx_claim_evidencias_evidencia
    ON claim_evidencias (evidencia_id);
