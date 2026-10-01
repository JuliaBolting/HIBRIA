"""Validação local, sem consultas pagas, banco de produção ou Ollama.
HIBRIA_PROJECT_ROOT pode apontar para o diretório ai-engine a validar.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.getenv('HIBRIA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
import json
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from datetime import datetime, timezone
from types import SimpleNamespace as N
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pipeline.security import public_http as http
from pipeline.security.redaction import redact, error_summary
from pipeline.analysis.factual_evidence import eligible
from pipeline.output.aggregator import Aggregator
from pipeline.analysis.similarity import SimilarityCalculator
from pipeline.output.explanation_generator import ExplanationGenerator as G
from pipeline.output.explanation_context import build_context
from pipeline.output.explanation_cache import fingerprint
from pipeline.pipeline import PipelineResult
from pipeline.persistence.analysis_repository import CachedAnalysis
from pipeline.persistence.rag_memory_service import RagMemoryService
from pipeline.retrieval.vector_store import VectorStore, Document
from pipeline.retrieval.vector_storage import active_paths
from pipeline.analysis.reputation.identity import SourceIdentityResolver
from pipeline.analysis.reputation.service import SourceReputationService
from pipeline.retrieval.search_providers.serp import SerpSearchProvider
from pipeline.retrieval.search_providers.tavily import TavilySearchProvider
from api.schemas.request import AnalyzeRequest
from api.routes import analyze as routes
from api import app as app_module


def req():
    return AnalyzeRequest(url='https://fonte.example/noticia', title='Texto do cliente', content='Conteúdo escolhido pelo cliente. '*10)


def evidence(text='A escola abriu.', trusted=True, score=.9, **kwargs):
    return N(text=text, source='fonte.example', url='https://fonte.example/'+str(uuid4()),
             evidence_id=str(uuid4()), retrieval_layer='web_search', source_type='web',
             trusted_source=trusted, similarity=score, published_at=None, metadata={}, **kwargs)


class HttpTests(unittest.TestCase):
    def test_private_and_credentials_blocked(self):
        for url in ['http://127.0.0.1/', 'http://169.254.169.254/', 'http://10.0.0.1/',
                    'http://[::1]/', 'http://[::ffff:127.0.0.1]/', 'http://100.64.0.1/',
                    'file:///etc/passwd', 'https://user:password@example.com/', 'http://localhost/',
                    'http://host.internal/', 'http://example.com:5432/']:
            with self.subTest(url=url), self.assertRaises(http.UnsafeURL):
                http.validate_url(url, resolve=False)

    def test_dns_rebinding_and_mixed_dns_blocked(self):
        for addresses in [['127.0.0.1'], ['93.184.216.34', '10.1.2.3']]:
            answers=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443)) for ip in addresses]
            with patch.object(http.socket,'getaddrinfo',return_value=answers), self.assertRaises(http.UnsafeURL):
                http.validate_url('https://news.example/')

    def test_dns_pinned_and_tls_hostname(self):
        raw=MagicMock(status=200,headers={'Content-Type':'text/plain; charset=utf-8'})
        raw.stream.return_value=[b'ok']
        with patch.object(http.socket,'getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]) as dns, \
             patch.object(http.urllib3,'HTTPSConnectionPool') as pool:
            pool.return_value.urlopen.return_value=raw
            response=http.get('https://news.example/test?q=abc',timeout=2)
            self.assertEqual(response.text,'ok')
            self.assertEqual(dns.call_count,1)
            self.assertEqual(pool.call_args.kwargs['host'],'93.184.216.34')
            self.assertEqual(pool.call_args.kwargs['server_hostname'],'news.example')
            self.assertEqual(pool.call_args.kwargs['cert_reqs'],'CERT_REQUIRED')
            self.assertEqual(pool.return_value.urlopen.call_args.args[1],'/test?q=abc')
            self.assertFalse(pool.return_value.urlopen.call_args.kwargs['redirect'])

    def test_redirect_to_metadata_blocked(self):
        raw=MagicMock(status=302,headers={'Location':'http://169.254.169.254/latest/'})
        with patch.object(http.socket,'getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]), \
             patch.object(http.urllib3,'HTTPSConnectionPool') as pool:
            pool.return_value.urlopen.return_value=raw
            with self.assertRaises(http.UnsafeURL):http.get('https://news.example/')
            self.assertEqual(pool.call_count,1)

    def test_cross_origin_redirect_strips_key_header(self):
        first=MagicMock(status=302,headers={'Location':'https://other.example/'})
        second=MagicMock(status=200,headers={});second.stream.return_value=[b'ok']
        headers=[]
        def send(*a,**kw):
            headers.append(dict(kw['headers']))
            return first if len(headers)==1 else second
        with patch.object(http.socket,'getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]), \
             patch.object(http.urllib3,'HTTPSConnectionPool') as pool:
            pool.return_value.urlopen.side_effect=send
            http.get('https://news.example/',headers={'Authorization':'secret-test','X-API-KEY':'secret-test'})
        self.assertNotIn('Authorization',headers[1]);self.assertNotIn('X-API-KEY',headers[1])

    def test_post_cross_origin_blocked(self):
        raw=MagicMock(status=307,headers={'Location':'https://other.example/'})
        with patch.object(http.socket,'getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]), \
             patch.object(http.urllib3,'HTTPSConnectionPool') as pool:
            pool.return_value.urlopen.return_value=raw
            with self.assertRaises(http.UnsafeURL):http.post('https://news.example/',json={'api_key':'dummy'})

    def test_response_size_limit(self):
        raw=MagicMock(status=200,headers={});raw.stream.return_value=[b'12345']
        with patch.object(http.socket,'getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]), \
             patch.object(http.urllib3,'HTTPSConnectionPool') as pool,patch.object(http,'MAX_BYTES',4):
            pool.return_value.urlopen.return_value=raw
            with self.assertRaises(http.RequestException):http.get('https://news.example/')

    def test_error_redaction(self):
        response=http.PublicResponse();response.status_code=401;response.url='https://api.example/?api_key=dummy-secret'
        with self.assertRaises(http.exceptions.HTTPError) as err:response.raise_for_status()
        self.assertNotIn('dummy-secret',str(err.exception))
        with patch.dict(os.environ,{'SERPAPI_API_KEY':'dummy-secret'}):
            payload=redact({'provider_failures':{'serp':'bad dummy-secret'},'text':'Error dummy-secret'})
        self.assertNotIn('dummy-secret',json.dumps(payload))
        self.assertNotIn('dummy-secret',error_summary(err.exception))

    def test_canonical_and_redirect_cannot_inherit_reputation(self):
        for final in ['https://unknown.example/','https://approved.example/']:
            response=N(status_code=200,history=[],url=final,headers={'Content-Type':'text/html'},
                       text='<html><link rel="canonical" href="https://approved.example/"></html>')
            with patch('pipeline.analysis.reputation.identity.requests.get',return_value=response):
                identity=SourceIdentityResolver().resolve('https://unknown.example/article')
                self.assertEqual(identity.canonical_domain,'unknown.example')
                self.assertEqual(identity.aliases,[])


class CalculationTests(unittest.TestCase):
    def test_untrusted_cannot_label_unreliable(self):
        ev=N(evidence_layer='web_search',source_type='web',trusted_source=False,is_sufficient=True,
             evidence_url='https://bad.example/',similarity_final=.9)
        sim=N(claim_id='a',has_evidence=True,score=.9,top_evidence=ev,evidences=[ev])
        stance=N(claim_id='a',url=ev.evidence_url,stance='contradict',similarity=.9,trusted_source=False)
        result=N(similarity_scores=[sim],stance_results=[stance],reputation=None,classification=None)
        output=Aggregator.aggregate(result)
        self.assertNotEqual(output['label'],'não confiável')
        self.assertEqual(output['breakdown']['stance_stats']['contradict'],0)

    def test_valid_evidence_is_not_hidden(self):
        claim=N(claim_id='a',text='A escola foi inaugurada.',normalized='A escola foi inaugurada.')
        bad=evidence('Texto não aprovado.',False,.95);good=evidence('Texto aprovado.',True,.85)
        embed=N(embed_batch=lambda *a,**kw:np.array([[1.,0.],[1.,0.],[.9,np.sqrt(.19)]],dtype=np.float32))
        with patch.object(SimilarityCalculator,'_get_model',return_value=embed):
            result=SimilarityCalculator.calculate([claim],[N(claim=claim,evidences=[bad,good])])[0]
        self.assertTrue(result.top_evidence.trusted_source)
        self.assertTrue(Aggregator._is_valid_factual_evidence(result))

    def test_eligible_filters_corpus_and_context(self):
        for ev in [N(retrieval_layer='wikipedia',trusted_source=True,similarity=.99),
                   N(source='Fake.Br/true',trusted_source=True,similarity=.99),
                   N(url='fakebr://true/1',trusted_source=True,similarity=.99)]:
            self.assertFalse(eligible(ev))

    def test_explicit_false_wins(self):
        with patch.dict(os.environ,{'SERPER_API_KEY':'dummy-valid-key','SERPAPI_API_KEY':'','SEARCHAPI_API_KEY':'',
                           'HIBRIA_ENABLE_SERPER':'false','HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS':'true',
                           'TAVILY_API_KEY':'dummy-valid-key','HIBRIA_ENABLE_TAVILY':'false'}):
            self.assertEqual(SerpSearchProvider()._provider_candidates(),[])
            self.assertFalse(TavilySearchProvider().enabled)

    def test_inverted_explanation_rejected(self):
        claim=N(claim_id='school',text='A prefeitura anunciou uma escola.')
        other=N(claim_id='library',text='A prefeitura anunciou uma biblioteca.')
        result=N(title='Escola e biblioteca do município',label_final='evidência insuficiente',score_final=42,
             score_breakdown={'coverage_score':10,'reputation_score':95,'reputation_status':'evaluated'},
             retrieval_results=[N(claim=claim,evidences=[evidence(stance='support')]),
                                N(claim=other,evidences=[evidence(stance='insufficient')])],stance_results=[])
        inverted={'explanation':'O resultado foi evidência insuficiente porque a biblioteca recebeu confirmação, mas a escola não teve apoio suficiente.',
                  'details':['A biblioteca foi confirmada pelos materiais consultados durante a análise.',
                             'A escola ficou sem confirmação suficiente nas verificações realizadas.',
                             'A boa reputação do veículo contribuiu para a nota, sem resolver as dúvidas restantes.']}
        report=G._validate_model_report(inverted,result)
        self.assertFalse(report['validation']['accepted']['explanation'])
        self.assertFalse(report['validation']['accepted']['details.0'])
        self.assertFalse(report['validation']['accepted']['details.1'])
        correct={**inverted,'explanation':'O resultado foi evidência insuficiente porque a escola recebeu confirmação, mas a biblioteca ficou sem apoio suficiente.',
                 'details':['A escola recebeu apoio nos materiais examinados durante a análise.',
                            'A biblioteca ficou sem confirmação suficiente nas verificações realizadas.',inverted['details'][2]]}
        valid=G._validate_model_report(correct,result)
        self.assertTrue(valid['validation']['accepted']['explanation'],valid)
        self.assertTrue(valid['validation']['accepted']['details.0'],valid)

    def test_rejected_source_not_support_in_context(self):
        result=N(retrieval_results=[N(claim=N(claim_id='a',text='A biblioteca abriu.'),
                 evidences=[evidence(trusted=False,stance='support')])])
        context=build_context(result)
        self.assertEqual(G._point_groups(context)['support'],[])


class CacheAndApiTests(unittest.TestCase):
    def test_old_unverified_cache_not_used(self):
        repo=MagicMock();repo.get.return_value=CachedAnalysis('id',self.payload(),None,1)
        self.assertIsNone(routes._cached_data(repo,req()))

    @staticmethod
    def payload():
        return {'analysis':{'score':42,'label':'evidência insuficiente','title':'Escola'},
                'explanation':'Explicação salva.','details':[],'metadata':{}}

    def test_refresh_only_explanation(self):
        data=self.payload();data['metadata']={'input_policy':routes.CACHE_POLICY,'explanation_version':'old'}
        repo=MagicMock();repo.get.return_value=CachedAnalysis('id',data,datetime.now(timezone.utc),2)
        def updated(value):
            value['explanation']='Texto atualizado com os mesmos dados.'
            value['metadata']['explanation_fingerprint']=fingerprint()
            return value
        with patch.object(routes,'refresh',side_effect=updated) as model,patch.object(routes.HibriaPipeline,'run') as pipeline:
            self.assertIsNone(routes._cached_data(repo,req()))
            answer=routes._cached_data(repo,req(),refresh_explanation=True)
            self.assertEqual(model.call_count,1);pipeline.assert_not_called()
        self.assertEqual(answer['explanation'],'Texto atualizado com os mesmos dados.')
        self.assertTrue(answer['metadata']['cache']['hit']);repo.revise_explanation.assert_called_once()

    def test_verified_cache_is_fast(self):
        data=self.payload();data['metadata']={'input_policy':routes.CACHE_POLICY,'explanation_fingerprint':fingerprint()}
        repo=MagicMock();repo.get.return_value=CachedAnalysis('id',data,None,2)
        with patch.object(routes,'refresh') as generate:
            self.assertIsNotNone(routes._cached_data(repo,req()))
            generate.assert_not_called()

    def test_client_content_is_not_used_for_pipeline_or_save(self):
        repo=MagicMock();repo.get.return_value=None;repo.save.return_value='saved'
        saved=self.payload()
        result=N(response=saved,title='Título conferido',score_final=42,label_final='evidência insuficiente',
                 explanation=saved['explanation'],_processing_time={})
        server_text='Conteúdo conferido no servidor. '*10
        with patch.object(routes,'AnalysisRepository',return_value=repo), \
             patch.object(routes.TextExtractor,'extract',return_value={'title':'Título conferido','content':server_text}), \
             patch.object(routes.HibriaPipeline,'run',return_value=result) as run, \
             patch.object(routes.HibriaPipeline,'_get_vector_store',return_value=None), \
             patch.object(routes,'RagMemoryService') as memory:
            memory.return_value.persist_and_index.return_value={}
            routes._perform_analysis(req())
        self.assertEqual(run.call_args.kwargs['content'],server_text.strip())
        self.assertEqual(repo.save.call_args.kwargs['content'],server_text.strip())
        self.assertEqual(repo.save.call_args.kwargs['data']['metadata']['input_policy'],routes.CACHE_POLICY)

    def test_failed_verification_does_not_save(self):
        repo=MagicMock();repo.get.return_value=None
        with patch.object(routes,'AnalysisRepository',return_value=repo), \
             patch.object(routes.TextExtractor,'extract',return_value={'content':'curto'}), \
             self.assertRaises(routes.ExtractionError):
            routes._perform_analysis(req())
        repo.save.assert_not_called()

    def test_queue_bounded_and_idempotent(self):
        executor=MagicMock();futures=[]
        def submit(*args):
            future=Future();futures.append(future);return future
        executor.submit.side_effect=submit
        with patch.object(routes,'_jobs',{}),patch.object(routes,'_job_executor',executor), \
             patch.object(routes,'_job_admission',threading.BoundedSemaphore(2)):
            key=uuid4();routes.start_analysis_job(key,req());routes.start_analysis_job(key,req())
            self.assertEqual(len(futures),1)
            routes.start_analysis_job(uuid4(),req())
            with self.assertRaises(HTTPException) as err:routes.start_analysis_job(uuid4(),req())
            self.assertEqual(err.exception.status_code,429)
            routes.cancel_analysis_job(key)
            routes.start_analysis_job(uuid4(),req())
            self.assertEqual(len(futures),3)

    def test_deadline_starts_before_execution(self):
        with self.assertRaises(HTTPException) as err:routes._acquire_analysis_slot(None,time.monotonic()-1)
        self.assertEqual(err.exception.status_code,503)

    def test_api_private_url_and_body_size(self):
        with patch.object(app_module,'HIBRIA_PUBLIC_API',True),patch.object(app_module,'HIBRIA_API_KEY',''):
            client=TestClient(app_module.app)
            answer=client.post('/analyze',json={'url':'http://169.254.169.254/','content':'x'*120})
            self.assertEqual(answer.status_code,422,answer.text)
            huge=client.post('/analyze',content='x'*1_500_001,headers={'Content-Type':'application/json'})
            self.assertEqual(huge.status_code,413,huge.text)

    def test_feedback_rate_limit(self):
        from api.routes import feedback
        with patch.object(app_module,'HIBRIA_PUBLIC_API',True),patch.object(app_module,'HIBRIA_API_KEY',''), \
             patch.object(app_module,'_rate_windows',__import__('collections').defaultdict(__import__('collections').deque)), \
             patch.object(feedback,'FeedbackRepository') as repository:
            repository.return_value.save.return_value=N(rating=5,category='positiva',already_submitted=False)
            repository.return_value.summary.return_value={}
            client=TestClient(app_module.app)
            answers=[client.post('/feedback',json={'analysis_id':str(uuid4()),'evaluator_id':str(uuid4()),'rating':5})
                     for _ in range(app_module.RATE_LIMIT_PER_MINUTE+1)]
        self.assertEqual(answers[-1].status_code,429)
        self.assertEqual(repository.return_value.save.call_count,app_module.RATE_LIMIT_PER_MINUTE)


class Embedding:
    dims=2
    def embed_batch(self,texts,**kwargs):return np.tile(np.array([[1.,0.]],dtype=np.float32),(len(texts),1))
    def embed(self,text,**kwargs):return np.array([1.,0.],dtype=np.float32)


class FaissTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.embed=patch('pipeline.analysis.embeddings.EmbeddingModel',return_value=Embedding())
        self.embed.start();self.addCleanup(self.embed.stop)
        self.store=VectorStore(self.temp.name)
        self.store.add_document(Document(text='Documento inicial.',source='Fonte',url='https://fonte.example/1',doc_id='initial'))

    def test_failed_write_never_false_reconciles(self):
        candidate=N(evidence_id='e2',text='Outro documento que deve ser persistido.',source='Fonte',url='https://fonte.example/2',published_at=None,metadata={})
        repo=MagicMock();service=RagMemoryService(repo)
        summary={'indexed_evidences':0,'reconciled_evidences':0,'failed_indexing':0}
        with patch.object(self.store._get_faiss(),'write_index',side_effect=OSError('simulado')):
            service._index_candidates([candidate],self.store,summary)
        self.assertFalse(self.store.contains_document('rag-evidence-e2'))
        repo.mark_indexed.assert_not_called()
        service._index_candidates([candidate],self.store,summary)
        self.assertTrue(VectorStore(self.temp.name).contains_document('rag-evidence-e2'))
        self.assertEqual(summary['reconciled_evidences'],0)
        self.assertEqual(summary['indexed_evidences'],1)

    def test_manifest_failure_preserves_previous_pair(self):
        before=active_paths(self.temp.name)
        with patch('pipeline.retrieval.vector_storage.os.replace',side_effect=OSError('simulado')):
            with self.assertRaises(OSError):self.store.add_document(Document(text='Não salvo.',source='Fonte',url='',doc_id='failed'))
        self.assertEqual(active_paths(self.temp.name),before)
        reloaded=VectorStore(self.temp.name)
        self.assertTrue(reloaded.contains_document('initial'));self.assertFalse(reloaded.contains_document('failed'))

    def test_two_store_instances_do_not_overwrite_each_other(self):
        other=VectorStore(self.temp.name)
        self.store.add_document(Document(text='Segundo documento.',source='Fonte',url='',doc_id='second'))
        other.add_document(Document(text='Terceiro documento.',source='Fonte',url='',doc_id='third'))
        reloaded=VectorStore(self.temp.name)
        for item in ['initial','second','third']:self.assertTrue(reloaded.contains_document(item))

    def test_corrupt_pair_is_not_silently_reset(self):
        _,metadata=active_paths(self.temp.name);metadata.write_text('[]')
        with self.assertRaises(ValueError):VectorStore(self.temp.name)


if __name__=='__main__':unittest.main(verbosity=2)
