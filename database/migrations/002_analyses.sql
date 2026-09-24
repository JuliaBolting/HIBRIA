-- =============================================================================
-- Cache e histórico das análises da HÍBRIA
--
-- Uma análise recente é reutilizada quando coincidirem:
--   - URL normalizada;
--   - versão do pipeline.
-- O hash do conteúdo permanece armazenado para histórico, auditoria e para
-- distinguir versões da página após a expiração do cache. Ele não participa da
-- consulta porque páginas jornalísticas possuem trechos dinâmicos que podem
-- mudar entre duas capturas da mesma notícia.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
