-- =============================================================================
-- Avaliação anônima dos resultados da HÍBRIA
--
-- Escala usada no teste com usuários:
--   1 e 2 estrelas -> avaliação negativa
--   3 estrelas     -> avaliação neutra
--   4 e 5 estrelas -> avaliação positiva
--
-- O identificador é aleatório e diferente para cada análise avaliada. Nenhum
-- nome, e-mail, IP ou outro dado pessoal é armazenado nesta tabela, e avaliações
-- distintas não podem ser associadas entre si pelo identificador.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS avaliacoes_analise (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analise_id UUID NOT NULL REFERENCES analises(id) ON DELETE CASCADE,
    avaliador_id UUID NOT NULL,
    nota SMALLINT NOT NULL,
    categoria VARCHAR(12) GENERATED ALWAYS AS (
        CASE
            WHEN nota >= 4 THEN 'positiva'
            WHEN nota = 3 THEN 'neutra'
            ELSE 'negativa'
        END
    ) STORED,
    data_avaliacao TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_avaliacoes_analise_usuario
        UNIQUE (analise_id, avaliador_id),
    CONSTRAINT ck_avaliacoes_analise_nota
        CHECK (nota BETWEEN 1 AND 5)
);

CREATE INDEX IF NOT EXISTS idx_avaliacoes_analise_id
    ON avaliacoes_analise (analise_id);

CREATE INDEX IF NOT EXISTS idx_avaliacoes_categoria
    ON avaliacoes_analise (categoria);

CREATE OR REPLACE VIEW resumo_avaliacoes_por_analise AS
SELECT
    analise_id,
    COUNT(*)::INTEGER AS total_avaliacoes,
    ROUND(AVG(nota), 2) AS media,
    COUNT(*) FILTER (WHERE categoria = 'positiva')::INTEGER AS positivas,
    COUNT(*) FILTER (WHERE categoria = 'neutra')::INTEGER AS neutras,
    COUNT(*) FILTER (WHERE categoria = 'negativa')::INTEGER AS negativas
FROM avaliacoes_analise
GROUP BY analise_id;

CREATE OR REPLACE VIEW resumo_avaliacoes_sistema AS
SELECT
    COUNT(*)::INTEGER AS total_avaliacoes,
    ROUND(AVG(nota), 2) AS media_geral,
    COUNT(*) FILTER (WHERE categoria = 'positiva')::INTEGER AS positivas,
    COUNT(*) FILTER (WHERE categoria = 'neutra')::INTEGER AS neutras,
    COUNT(*) FILTER (WHERE categoria = 'negativa')::INTEGER AS negativas,
    COALESCE(
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE categoria = 'positiva')
            / NULLIF(COUNT(*), 0),
            2
        ),
        0
    ) AS percentual_positivas,
    COALESCE(
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE categoria = 'neutra')
            / NULLIF(COUNT(*), 0),
            2
        ),
        0
    ) AS percentual_neutras,
    COALESCE(
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE categoria = 'negativa')
            / NULLIF(COUNT(*), 0),
            2
        ),
        0
    ) AS percentual_negativas,
    MAX(data_avaliacao) AS atualizada_em
FROM avaliacoes_analise;
