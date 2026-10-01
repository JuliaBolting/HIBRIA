import os,sys,unittest,threading,tempfile,signal
from pathlib import Path
sys.path.insert(0,os.getenv('HIBRIA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
from unittest.mock import patch,MagicMock
from types import SimpleNamespace as N
from fastapi.testclient import TestClient
from api import app as appmod
from api.routes import analyze as route
from api.schemas.feedback import FeedbackRequest
from pipeline.cancellation import cancellable,AnalysisCancelled
from pipeline.retrieval.vector_store import VectorStore
from pipeline.retrieval.retriever import EvidenceRetriever
from pipeline.retrieval.search_providers.quota import ProviderQuota
from pipeline.retrieval.source_trust import SourceTrustResolver
from pipeline.persistence.analysis_repository import AnalysisRepository,PersistenceError
from pipeline.analysis.claim_detector import Claim
from pipeline.security import public_http
from uuid import uuid4

class Controls(unittest.TestCase):
 def test_host_cannot_bypass_auth(self):
  with patch.object(appmod,'HIBRIA_API_KEY','testkey'),patch.object(appmod,'HIBRIA_PUBLIC_API',False),patch.object(route,'_perform_analysis') as run:
   c=TestClient(appmod.app)
   for host in ['example.com/abc?bar=','example.com#','example.com\\path','example.com@other']:
    self.assertEqual(c.post('/analyze',json={},headers={'host':host}).status_code,400)
   self.assertEqual(c.post('/analyze',json={}).status_code,401)
   run.assert_not_called()
 def test_chunking_progress_and_tail(self):
  store=object.__new__(VectorStore)
  for text in ['a'*100+' '+'b'*1000,'x'*4000,'one two three '*300,'fim', 'a'*900]:
   # Timeout detects regressions without letting an infinite loop allocate memory.
   old=signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError()))
   signal.setitimer(signal.ITIMER_REAL,1)
   try: chunks=store._chunk_text(text)
   finally: signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
   self.assertLess(len(chunks),len(text)+1)
   self.assertTrue(chunks[-1].endswith(text.strip()[-20:]))
   self.assertTrue(all(len(s)<=store.MAX_CHUNK_CHARS for s in chunks))
 def test_cancel_stops_next_provider_and_clears_context(self):
  event=threading.Event();calls=[]
  r=EvidenceRetriever()
  def first(*a,**k): calls.append('first');event.set();return []
  def second(*a,**k): calls.append('second');return []
  r._sources=[N(name='first',is_available=lambda:True,search=first),N(name='second',is_available=lambda:True,search=second)]
  @cancellable
  def run(*,cancel_event): return r.retrieve(Claim(text='Hospital abriu centro',normalized='',claim_id='c',subject=''))
  with self.assertRaises(AnalysisCancelled): run(cancel_event=event)
  self.assertEqual(calls,['first'])
  from pipeline.cancellation import checkpoint
  checkpoint() # Não contamina a próxima análise no mesmo worker.
 def test_cancel_does_not_reserve_api_credit(self):
  event=threading.Event()
  @cancellable
  def run(*,cancel_event):
   event.set();ProviderQuota.try_register('serper')
  with patch.object(ProviderQuota,'_write') as write,self.assertRaises(AnalysisCancelled): run(cancel_event=event)
  write.assert_not_called()
 def test_cancel_does_not_start_network(self):
  event=threading.Event()
  @cancellable
  def run(*,cancel_event): event.set();public_http.get('https://example.com')
  with patch.object(public_http,'validate_url') as validate,self.assertRaises(AnalysisCancelled):run(cancel_event=event)
  validate.assert_not_called()
 def test_half_stars_only(self):
  params={'analysis_id':uuid4(),'evaluator_id':uuid4()}
  for rating in [0.5,1,2.5,3,3.5,4.5,5]: self.assertEqual(FeedbackRequest(**params,rating=rating).rating,rating)
  for rating in [0,0.7,3.1,5.5,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):FeedbackRequest(**params,rating=rating)
 def test_no_database_fails_before_paid_work(self):
  with self.assertRaises(PersistenceError):AnalysisRepository(database_url='').ensure_ready()
 def test_pipeline_cache_version_changes_even_with_old_env(self):
  with patch.dict(os.environ,{'HIBRIA_PIPELINE_VERSION':'1.2.1'}):self.assertIn('directional-2',AnalysisRepository.pipeline_version())
 def test_broken_required_step_is_not_a_finished_analysis(self):
  from contextlib import ExitStack
  from pipeline.pipeline import HibriaPipeline as P
  with ExitStack() as stack:
   stack.enter_context(patch.dict(os.environ,{'HIBRIA_SHOW_PIPELINE_PROGRESS':'false'}))
   stack.enter_context(patch.object(P,'_step_extract',side_effect=lambda url,r,**kwargs:r))
   for name in ['_step_clean','_step_normalize','_step_segment']:
    stack.enter_context(patch.object(P,name,side_effect=lambda r:r))
   stack.enter_context(patch.object(P,'_step_detect_claims',side_effect=ValueError('invalid')))
   retrieve=stack.enter_context(patch.object(P,'_step_retrieve'))
   with self.assertRaisesRegex(RuntimeError,'claim_detector'):P.run('https://news.example/item')
   retrieve.assert_not_called()
 def test_score_direction_and_absence(self):
  from pipeline.output.aggregator import Aggregator
  def calculate(stance):
   e=N(evidence_id='e',evidence_url='https://reference.example/n',evidence_layer='web_search',
       source_type='news',trusted_source=True,similarity_final=.95,is_sufficient=True,metadata={})
   result=N(similarity_scores=[N(claim_id='c',score=.95,has_evidence=True,top_evidence=e,evidences=[e])],
       stance_results=[N(claim_id='c',evidence_id='e',stance=stance)],reputation={'status':'evaluated','score':.95},classification={'status':'ok','score':.99})
   return Aggregator.aggregate(result)
  support,neutral,against=map(calculate,['support','neutral','contradict'])
  self.assertGreater(support['score'],neutral['score'])
  self.assertGreater(neutral['score'],against['score'])
  self.assertEqual(support['label'],'confiável')
  self.assertEqual(neutral['label'],'evidência insuficiente')
  self.assertEqual(against['label'],'não confiável')
 def test_database_preflight_stops_before_extraction(self):
  from api.schemas.request import AnalyzeRequest
  repo=MagicMock();repo.ensure_ready.side_effect=PersistenceError('offline')
  with patch.object(route,'AnalysisRepository',return_value=repo),patch.object(route.TextExtractor,'extract') as extract:
   with self.assertRaises(PersistenceError):route._perform_analysis(AnalyzeRequest(url='https://news.example/item',content='texto '*30))
   extract.assert_not_called()
 def test_dynamic_reference_budget_and_cache(self):
  conn=MagicMock();conn.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value=None
  assessed=N(status='evaluated',score=.9,identity=N(canonical_domain='one.example'))
  with patch.dict(os.environ,{'HIBRIA_REFERENCE_REPUTATION_BUDGET':'1'}),patch('psycopg2.connect',return_value=conn),patch('pipeline.analysis.reputation.config.DYNAMIC_ENABLED',True),patch('pipeline.analysis.reputation.service.SourceReputationService') as service:
   service.return_value.get_or_evaluate.return_value=assessed
   resolver=SourceTrustResolver(database_url='fake')
   self.assertEqual(resolver.resolve('one.example'),(True,.9))
   self.assertEqual(resolver.resolve('one.example'),(True,.9))
   self.assertEqual(resolver.resolve('two.example'),(False,None))
   service.return_value.get_or_evaluate.assert_called_once()
 def test_dynamic_disabled_does_not_call_providers(self):
  conn=MagicMock();conn.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value=None
  with patch('psycopg2.connect',return_value=conn),patch('pipeline.analysis.reputation.config.DYNAMIC_ENABLED',False),patch('pipeline.analysis.reputation.service.SourceReputationService') as service:
   self.assertEqual(SourceTrustResolver(database_url='fake').resolve('one.example'),(False,None))
   service.assert_not_called()

if __name__=='__main__':unittest.main()
