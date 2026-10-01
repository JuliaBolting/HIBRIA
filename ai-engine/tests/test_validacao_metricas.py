"""Casos artificiais para conferir as contas do avaliador; não são notícias reais."""
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location('avaliar_hibria_offline', Path(__file__).resolve().parents[1]/'scripts/avaliar_hibria_offline.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def row(i, truth, prediction):
    return {'id':str(i), 'referencia':truth, 'classificacao_final':prediction,
            'versao_pipeline':'test-version', 'url':f'https://example.test/{i}'}


def label(i, truth='verdadeira', **changes):
    return dict({'id':str(i), 'referencia':truth, 'fonte_referencia':'documento independente',
                 'justificativa':'caso artificial', 'avaliador':'A', 'data_rotulagem':'2026-10-01',
                 'usada_no_ajuste':'nao'}, **changes)


class OfflineMetricsTests(unittest.TestCase):
    def test_abstentions_are_not_false(self):
        rows=[row(1,'verdadeira','confiável'),row(2,'verdadeira','evidência insuficiente'),
              row(3,'falsa','não confiável'),row(4,'falsa','confiável'),
              row(5,'verdadeira','não confiável'),row(6,'falsa','parcialmente confiável')]
        r=m.metrics_for(rows)
        self.assertEqual((r['tp'],r['tn'],r['fp'],r['fn']),(1,1,1,1))
        self.assertEqual(r['abstencoes'],2)
        self.assertEqual(r['acuracia_seletiva'],.5)
        self.assertEqual(r['acertos_sobre_todas_binarias'],2/6)
        self.assertEqual(r['cobertura_de_decisao'],4/6)
        self.assertEqual(r['macro_f1_decididas'],.5)
    def test_empty_is_not_perfect(self):
        r=m.metrics_for([])
        self.assertIsNone(r['acuracia_seletiva'])
        self.assertIsNone(r['ic95_wilson_acuracia_seletiva'])
    def test_only_abstentions_no_accuracy(self):
        r=m.metrics_for([row(1,'falsa','evidência insuficiente')])
        self.assertEqual(r['cobertura_de_decisao'],0)
        self.assertIsNone(r['acuracia_seletiva'])
        self.assertEqual(r['recall_falsas_incluindo_abstencoes'],0)
    def test_inconclusive_reference_not_binary(self):
        r=m.metrics_for([row(1,'inconclusiva','confiável')])
        self.assertEqual(r['binarias'],0)
        self.assertEqual(r['rotuladas'],1)
    def test_missing_class_no_balanced_accuracy(self):
        r=m.metrics_for([row(1,'verdadeira','confiável')])
        self.assertIsNone(r['acuracia_balanceada_decididas'])
    def test_wilson_known_case(self):
        low,high=m.wilson(50,100)
        self.assertAlmostEqual(low,.40383153,6)
        self.assertAlmostEqual(high,.59616847,6)
    def test_perfect_sample_has_uncertainty(self):
        self.assertLess(m.wilson(3,3)[0],.5)
        self.assertAlmostEqual(m.wilson(0,3)[0],0)
    def test_duplicate_url_or_content_removed(self):
        items=[dict(row(1,'verdadeira','confiável'),hash_conteudo='a'),
               dict(row(2,'verdadeira','confiável'),hash_conteudo='a'),
               dict(row(1,'verdadeira','confiável'),hash_conteudo='b')]
        self.assertEqual(len(m.deduplicate(items)),1)
    def test_tuning_excluded_by_default(self):
        r,_=m.evaluate([row(1,'verdadeira','confiável')],[label(1,usada_no_ajuste='sim')])
        self.assertEqual(r['binarias'],0)
        self.assertEqual(r['excluidas']['usada_no_ajuste'],1)
    def test_duplicate_label_fails(self):
        with self.assertRaises(ValueError):m.evaluate([row(1,'verdadeira','confiável')],[label(1),label(1)])
    def test_missing_provenance_fails(self):
        with self.assertRaises(ValueError):m.evaluate([row(1,'verdadeira','confiável')],[label(1,fonte_referencia='')])
    def test_group_republication_fails(self):
        with self.assertRaises(ValueError):
            m.evaluate([row(i,'verdadeira','confiável') for i in [1,2]],
                       [label(i,grupo_noticia='mesmo-evento') for i in [1,2]])
    def test_versions_are_separate(self):
        second=dict(row(2,'falsa','confiável'),versao_pipeline='old')
        r,_=m.evaluate([row(1,'verdadeira','confiável'),second],[label(1),label(2,'falsa')])
        self.assertEqual(r['por_versao']['test-version']['acuracia_seletiva'],1)
        self.assertEqual(r['por_versao']['old']['acuracia_seletiva'],0)
    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):m.evaluate([row(1,'verdadeira','boa')],[label(1)])
    def test_percentiles_interpolate_and_ignore_missing(self):
        r=m.percentiles([None,10,20,30])
        self.assertEqual(r['p50'],20)
        self.assertEqual(r['p95'],29)


class OfflineReadOnlyTests(unittest.TestCase):
    def test_connection_sets_readonly_and_repeatable_read(self):
        from unittest.mock import patch, MagicMock
        conn=MagicMock()
        with patch('dotenv.dotenv_values',return_value={'HIBRIA_DATABASE_URL':'test-dsn'}), patch('psycopg2.connect',return_value=conn) as connect:
            self.assertIs(m.connect_readonly('/fake/env'),conn)
            connect.assert_called_once_with('test-dsn',connect_timeout=10)
            conn.set_session.assert_called_once_with(readonly=True,isolation_level='REPEATABLE READ')

    def test_report_escapes_article_html(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            m.save_report(tmp,'teste',{'title':'<script>alert(1)</script>'},'<img src=x>')
            page=(Path(tmp)/'teste.html').read_text()
            self.assertNotIn('<script>',page)
            self.assertNotIn('<img src=x>',page)

    def test_export_does_not_put_prediction_in_blind_labels(self):
        from unittest.mock import patch, MagicMock
        from types import SimpleNamespace as N
        import tempfile, csv, json
        conn=MagicMock();cur=conn.cursor.return_value.__enter__.return_value
        sample=dict(row(1,'verdadeira','confiável'),titulo='Exemplo',conteudo='Conteúdo',
                    url_normalizada='https://example.test/1',quantidade_consultas=1,
                    tempo_processamento_segundos=20)
        cur.fetchall.side_effect=[[sample],[]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'connect_readonly',return_value=conn), patch.object(m,'provenance',return_value={}):
            dest=Path(tmp)/'export'
            self.assertEqual(m.export_database(N(output=str(dest),env_file=None,pipeline_version=None,limit=0,seed=42)),0)
            with (dest/'rotulos.csv').open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);data=list(reader)
                self.assertNotIn('classificacao_final',reader.fieldnames)
                self.assertNotIn('score_final',reader.fieldnames)
                self.assertEqual(data[0]['referencia'],'')
            self.assertEqual(json.loads((dest/'analises.json').read_text())['analises'][0]['classificacao_final'],'confiável')
            for call in cur.execute.call_args_list:
                self.assertTrue(call.args[0].lstrip().upper().startswith(('SELECT','SET LOCAL')))
            conn.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
