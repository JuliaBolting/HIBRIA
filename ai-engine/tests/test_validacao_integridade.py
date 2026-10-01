import os
import sys
from pathlib import Path
sys.path.insert(0, os.getenv('HIBRIA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch, MagicMock
from uuid import uuid4
import threading

from pipeline.retrieval import retriever as r
from pipeline.analysis.claim_detector import Claim
from pipeline.analysis.factual_evidence import eligible
from pipeline.output.aggregator import Aggregator
from pipeline.output.explanation_context import build_context, item_group
from pipeline.output.explanation_generator import ExplanationGenerator as G
from pipeline.pipeline import HibriaPipeline
from api.routes import analyze as routes
from api.schemas.request import AnalyzeRequest

URL = 'https://jornal.example/noticia-hospital'
TITLE = 'Hospital municipal abriu um centro de atendimento para os moradores.'

def claim():
    return Claim(text=TITLE, normalized=TITLE, claim_id='c1', subject='Hospital municipal')

def ev(url=URL, trusted=True, score=.8, layer='web_search'):
    return r.Evidence(text=TITLE, source='jornal.example', url=url,
        similarity=score, claim_id='c1', trusted_source=trusted,
        retrieval_layer=layer, stance='support', metadata={'stance_similarity':score})

def result(evidence=None, failures=None):
    evidence = evidence or ev()
    return N(title=TITLE, score_final=60, label_final='parcialmente confiável',
       score_breakdown={'coverage_score':100, 'evidence_score':80,
           'reputation_score':90, 'reputation_status':'evaluated', 'bertimbau_score':80,
           'bertimbau_status':'ok', 'stance_stats':{'support':1},
           'weights':{'evidence':.6,'reputation':.25,'bertimbau':.15}},
       retrieval_results=[N(claim=claim(), evidences=[evidence], layers_failed=failures or {})],
       stance_results=[{'claim_id':'c1','evidence_id':evidence.evidence_id,
           'url':evidence.url,'stance':'support','similarity':.8}])

class Reproductions(unittest.TestCase):
    def test_00_strong_contradiction_reduces_reliability_score(self):
        evidence=N(evidence_id='e1',evidence_url='https://ref.example/1',
            evidence_layer='web_search',source_type='news',trusted_source=True,
            similarity_final=.95,is_sufficient=True,metadata={})
        data=N(similarity_scores=[N(claim_id='c1',has_evidence=True,score=.95,
            top_evidence=evidence,evidences=[evidence])],
            stance_results=[N(claim_id='c1',evidence_id='e1',stance='contradict')],
            reputation={'status':'evaluated','note':95,'score':.95},
            classification={'status':'ok','score':.99})
        output=Aggregator.aggregate(data)
        self.assertEqual(output['label'],'não confiável')
        self.assertLess(output['score'],40)
        print('\nReprodução do score:', output['score'], output['label'])

    def test_01_gdelt_cannot_return_article_itself(self):
        source=r.GdeltSource()
        retriever=r.EvidenceRetriever(current_url=URL)
        retriever._sources=[source]
        response=MagicMock(status_code=200,headers={'Content-Type':'application/json'},text='{}')
        response.json.return_value={'articles':[{'url':URL,'title':TITLE,'domain':'jornal.example'}]}
        with patch.object(source,'is_available',return_value=True), \
             patch.object(source,'_can_search',return_value=True), \
             patch.object(source,'_register_search',return_value=True), \
             patch.object(source,'_build_gdelt_queries',return_value=['hospital']), \
             patch.object(r.requests,'get',return_value=response), \
             patch.object(r,'_is_relevant_candidate',return_value=True), \
             patch.object(r,'_candidate_similarity',return_value=.9), \
             patch.object(retriever._trust_resolver,'resolve',return_value=(True,.9)), \
             patch.object(r.time,'sleep'):
            output=retriever.retrieve(claim())
        self.assertEqual(output.evidences, [])

    def test_02_top_k_preserves_eligible_evidence(self):
        bad=[ev(url=f'https://unknown.example/{i}',trusted=False,score=.99,layer='vector_store') for i in range(10)]
        good=ev(url='https://approved.example/1',trusted=True,score=.8)
        retriever=r.EvidenceRetriever()
        retriever._sources=[N(name='test',is_available=lambda:True,search=lambda *a,**k:bad+[good])]
        with patch.object(retriever._trust_resolver,'resolve',side_effect=lambda domain:(('approved.example' in domain),.9 if 'approved.example' in domain else None)):
            output=retriever.retrieve(claim(),top_k=10)
        self.assertTrue(eligible(good))
        self.assertIn(good,output.evidences)
        self.assertTrue(any(eligible(item) for item in output.evidences))

    def test_03_fallback_does_not_invent_a_limitation(self):
        context=build_context(result())
        self.assertEqual([item_group(i) for i in context['itens']],['support'])
        output=G.fallback_report(result())
        self.assertNotIn('ficou sem sustentação suficiente',output['explanation'])
        self.assertNotIn('permanece em aberto',output['details'][2])

    def test_04_disabled_provider_is_not_a_failure(self):
        data=result(failures={'newsapi':'não disponível (sem API key, limite atingido ou base local ausente)'})
        output=G.fallback_report(data)
        self.assertNotIn('Parte da busca não pôde ser concluída',output['details'][2])

    def test_05_database_save_failure_cannot_return_success(self):
        request=AnalyzeRequest(url=URL,content=TITLE*3)
        pipeline_result=N(title=TITLE,score_final=60,label_final='parcialmente confiável',
            explanation='Explicação de teste.',response={'analysis':{'score':60,'label':'parcialmente confiável'},
            'explanation':'Explicação de teste.','metadata':{}})
        repository=MagicMock()
        repository.get.return_value=None
        repository.save.return_value=None
        with patch.object(routes,'AnalysisRepository',return_value=repository), \
             patch.object(routes,'_cached_data',return_value=None), \
             patch.object(routes.TextExtractor,'extract',return_value={'content':TITLE*3,'title':TITLE}), \
             patch.object(routes.HibriaPipeline,'run',return_value=pipeline_result), \
             patch.object(routes.HibriaPipeline,'_get_vector_store',return_value=None), \
             patch.object(routes,'RagMemoryService') as memory:
            memory.return_value.persist_and_index.return_value={'error':'análise não persistida'}
            output=routes.analyze(request)
        self.assertFalse(output.success)
        self.assertIsNone(output.data)

    def test_06_unexpected_job_exception_is_logged(self):
        job_id=str(uuid4())
        request=AnalyzeRequest(url=URL,content=TITLE*3)
        with routes._jobs_lock:
            routes._jobs[job_id]=routes.AnalysisJob(job_id=job_id)
        try:
            with patch.object(routes,'_perform_analysis',side_effect=RuntimeError('erro simulado')), \
                 patch('logging.Logger._log') as log:
                routes._run_job(job_id,request)
            self.assertEqual(routes._jobs[job_id].status,'failed')
            log.assert_called_once()
        finally:
            with routes._jobs_lock: routes._jobs.pop(job_id,None)

if __name__=='__main__': unittest.main(verbosity=2)
