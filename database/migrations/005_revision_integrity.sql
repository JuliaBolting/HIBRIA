-- Histórico imutável por execução e avaliações de meia estrela.
-- Aplicar em transação. Mantém IDs, votos e resultados antigos.
ALTER TABLE analises DROP CONSTRAINT IF EXISTS uq_analises_cache;
ALTER TABLE analises ADD COLUMN IF NOT EXISTS revisao_criada_em TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE analises ADD COLUMN IF NOT EXISTS analise_anterior_id UUID REFERENCES analises(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_analises_cache_recente
    ON analises (hash_url, versao_pipeline, data_analise DESC, revisao_criada_em DESC);

-- Views e coluna gerada dependem do tipo de nota; são recriadas na transação.
DROP VIEW IF EXISTS resumo_avaliacoes_sistema;
DROP VIEW IF EXISTS resumo_avaliacoes_por_analise;
DROP INDEX IF EXISTS idx_avaliacoes_categoria;
ALTER TABLE avaliacoes_analise DROP COLUMN IF EXISTS categoria;
ALTER TABLE avaliacoes_analise DROP CONSTRAINT IF EXISTS ck_avaliacoes_analise_nota;
ALTER TABLE avaliacoes_analise ALTER COLUMN nota TYPE NUMERIC(2,1) USING nota::NUMERIC(2,1);
ALTER TABLE avaliacoes_analise ADD CONSTRAINT ck_avaliacoes_analise_nota
    CHECK (nota BETWEEN 0.5 AND 5 AND MOD(nota * 2, 1) = 0);
-- Mantém a categoria das notas inteiras: <3 negativa, 3 a 3,5 neutra, >=4 positiva.
ALTER TABLE avaliacoes_analise ADD COLUMN categoria VARCHAR(12) GENERATED ALWAYS AS (
    CASE WHEN nota >= 4 THEN 'positiva' WHEN nota >= 3 THEN 'neutra' ELSE 'negativa' END
) STORED;
CREATE INDEX idx_avaliacoes_categoria ON avaliacoes_analise(categoria);

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
