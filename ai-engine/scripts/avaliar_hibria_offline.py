#!/usr/bin/env python3
"""Validação reproduzível: testes isolados, carga de componentes e avaliação salva.
Não executa HibriaPipeline.run, consultas web, Ollama ou downloads de modelos.
Veja docs/VALIDACAO-OFFLINE.md. Python >= 3.10.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import unittest

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
VERSION = 'validacao-offline-1.0.0'
LABELS = ['confiável', 'não confiável', 'parcialmente confiável',
          'evidência insuficiente', 'não verificado']
TRUTHS = ('verdadeira', 'falsa', 'inconclusiva')
FIELDS = ['id', 'titulo', 'url', 'conteudo', 'referencia', 'fonte_referencia',
          'justificativa', 'avaliador', 'data_rotulagem', 'grupo_noticia',
          'usada_no_ajuste']


def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value).strip().lower())
                   if not unicodedata.combining(c))


def ratio(a, b):
    return a / b if b else None


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    margin = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-margin), min(1, center+margin)]


def percentiles(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) >= 0)
    def percentile(p):
        if not values:
            return None
        pos = (len(values)-1)*p
        lo = int(pos)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)
    return {'n': len(values), 'media': statistics.mean(values) if values else None,
            'p50': percentile(.5), 'p95': percentile(.95), 'maximo': max(values) if values else None}


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return 'indisponivel'
    from importlib.metadata import version, PackageNotFoundError
    dependencies = {}
    for name in ['fastapi', 'httpx', 'numpy', 'faiss-cpu', 'psycopg2-binary', 'python-dotenv']:
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = 'não instalada'
    return {'dependencias': dependencies, 'ferramenta': VERSION, 'instante_utc': datetime.now(timezone.utc).isoformat(),
            'commit': git('rev-parse', 'HEAD'), 'alteracoes_locais': git('status', '--porcelain'),
            'python': platform.python_version(), 'plataforma': platform.platform()}


def csv_safe(value):
    # Evita interpretar conteúdo de notícias como fórmula ao abrir em planilha.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_dir(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_report(directory, name, data, description):
    directory = Path(directory)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    (directory / (name+'.json')).write_text(serialized+'\n', encoding='utf-8')
    body = '<p>'+html.escape(description)+'</p>'
    if 'matriz' in data:
        body += '<table><caption>Referência independente × classificação da HÍBRIA</caption><tr><th>Referência</th>'
        body += ''.join('<th>'+html.escape(x)+'</th>' for x in LABELS)+ '</tr>'
        for truth, counts in data['matriz'].items():
            body += '<tr><th>'+truth+'</th>'+''.join('<td>'+str(counts[x])+'</td>' for x in LABELS)+'</tr>'
        body += '</table>'
    body += '<pre>'+html.escape(serialized)+'</pre>'
    page = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Validação HÍBRIA</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#15304a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd6e0;padding:10px;text-align:left}caption{font-weight:bold;margin:20px}</style><h1>Validação HÍBRIA</h1>'+body+'</html>'
    (directory / (name+'.html')).write_text(page, encoding='utf-8')
    print(directory / (name+'.html'))


@contextmanager
def isolated():
    """Bloqueio de rede/processos/BD, dotenv neutralizado e arquivos temporários.
    A trava por audit hook dura até o encerramento deste processo CLI.
    """
    from unittest.mock import patch
    import dotenv
    import psycopg2
    old = Path.cwd()
    keep = {k:v for k,v in os.environ.items() if not (
        k.startswith(('HIBRIA_', 'HF_', 'OLLAMA_', 'SERPER_', 'SERPAPI_', 'SEARCHAPI_', 'TAVILY_', 'BRAVE_', 'NEWSAPI_', 'GOOGLE_', 'GEMINI_', 'PG'))
        or k in {'DATABASE_URL', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'})}
    keep.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
                HIBRIA_PROJECT_ROOT=str(ENGINE), HIBRIA_SHOW_PIPELINE_PROGRESS='false')
    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'subprocess.Popen', 'os.system'}:
            raise RuntimeError('Operação externa bloqueada pela validação offline: '+event)
    sys.addaudithook(audit)
    sys.path.insert(0, str(ENGINE))
    with tempfile.TemporaryDirectory(prefix='hibria-testes-') as tmp, ExitStack() as stack:
        keep['HIBRIA_REPUTATION_JSON_PATH'] = str(Path(tmp)/'reputation.json')
        stack.enter_context(patch.dict(os.environ, keep, clear=True))
        stack.enter_context(patch.object(dotenv, 'load_dotenv', return_value=False))
        stack.enter_context(patch.object(psycopg2, 'connect', side_effect=RuntimeError('Banco bloqueado nos testes')))
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(old)


def run_tests(args):
    directory = output_dir(args.output)
    report = provenance()
    started = time.perf_counter()
    class Result(unittest.TextTestResult):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.items = []
        def startTest(self, test):
            self.begin = time.perf_counter()
            super().startTest(test)
        def record(self, test, state):
            self.items.append({'teste': test.id(), 'estado': state,
                               'segundos': round(time.perf_counter()-self.begin, 6)})
        def addSuccess(self, test):
            super().addSuccess(test); self.record(test, 'aprovado')
        def addFailure(self, test, err):
            super().addFailure(test, err); self.record(test, 'falhou')
        def addError(self, test, err):
            super().addError(test, err); self.record(test, 'erro')
        def addSkip(self, test, reason):
            super().addSkip(test, reason); self.record(test, 'ignorado: '+reason)
    with isolated(), (directory/'testes.log').open('w', encoding='utf-8') as log:
        suite = unittest.defaultTestLoader.discover(str(ENGINE/'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    report.update(testes=result.testsRun, falhas=len(result.failures), erros=len(result.errors),
                  ignorados=len(result.skipped), segundos=time.perf_counter()-started,
                  sucesso=result.wasSuccessful() and result.testsRun > 0 and not result.skipped,
                  casos=result.items, rede='bloqueada', banco='bloqueado', modelos='não executados')
    save_report(directory, 'testes', report, 'Testes de software com dados sintéticos e serviços simulados. Não medem acurácia em notícias reais.')
    print(f'{result.testsRun} testes; {len(result.failures)} falhas; {len(result.errors)} erros; {len(result.skipped)} ignorados.')
    return 0 if report['sucesso'] else 1


def deduplicate(rows):
    """Mantém a primeira (exportação ordenada por recência); agrupa URL OU texto."""
    seen_url, seen_content, selected = set(), set(), []
    for row in rows:
        url = row.get('url_normalizada') or row.get('url') or row['id']
        content = row.get('hash_conteudo')
        if not content and row.get('conteudo'):
            content = hashlib.sha256(row['conteudo'].encode()).hexdigest()
        if url in seen_url or (content and content in seen_content):
            continue
        seen_url.add(url)
        if content:
            seen_content.add(content)
        selected.append(row)
    return selected


def connect_readonly(env_file):
    import psycopg2
    from dotenv import dotenv_values
    values = dotenv_values(env_file) if env_file else os.environ
    url = values.get('HIBRIA_DATABASE_URL') or values.get('DATABASE_URL')
    if not url:
        raise ValueError('Configure a URL do banco ou informe --env-file com o arquivo do serviço.')
    connection = psycopg2.connect(url, connect_timeout=10)
    connection.set_session(readonly=True, isolation_level='REPEATABLE READ')
    return connection


def export_database(args):
    from psycopg2.extras import RealDictCursor
    directory = output_dir(args.output)
    report = provenance()
    connection = connect_readonly(args.env_file)
    try:
        with connection, connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = '30s'")
            cur.execute('''SELECT id::text, titulo, url_original AS url, url_normalizada,
                conteudo, hash_conteudo, versao_pipeline, score_final, classificacao_final,
                data_analise, revisao_criada_em, tempo_processamento_segundos,
                quantidade_consultas, resultado_json #>> '{metadata,explanation_source}' AS explanation_source,
                resultado_json #>> '{metadata,input_policy}' AS input_policy
                FROM analises ORDER BY data_analise DESC, revisao_criada_em DESC, id DESC''')
            rows = [dict(r) for r in cur.fetchall()]
            cur.execute('SELECT * FROM resumo_avaliacoes_sistema')
            feedback = [dict(r) for r in cur.fetchall()]
    finally:
        connection.close()
    versions = Counter(r['versao_pipeline'] for r in rows)
    if args.pipeline_version:
        rows = [r for r in rows if r['versao_pipeline'] == args.pipeline_version]
    eligible_count = len(rows)
    rows = deduplicate(rows)
    unique_count = len(rows)
    if args.limit and len(rows) > args.limit:
        # Sorteio reprodutível; não escolhe resultados bons pelo score.
        rows = random.Random(args.seed).sample(rows, args.limit)
    serialized = json.loads(json.dumps(rows, default=str, ensure_ascii=False))
    snapshot = {'schema': VERSION, 'proveniencia': report, 'selecao': {
        'versoes_no_banco': versions, 'filtro_versao': args.pipeline_version,
        'registros_antes_deduplicacao': eligible_count, 'noticias_unicas': unique_count,
        'exportadas': len(rows), 'semente': args.seed}, 'analises': serialized}
    (directory/'analises.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (directory/'rotulos.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:csv_safe(row.get(k, '')) for k in ['id', 'titulo', 'url', 'conteudo']})
    report.update(selecao=snapshot['selecao'], feedback=feedback,
                  classificacoes=dict(Counter(r['classificacao_final'] for r in rows)),
                  latencia_historica_segundos=percentiles(r['tempo_processamento_segundos'] for r in rows),
                  consultas_registradas=sum(r['quantidade_consultas'] for r in rows),
                  observacao='Latência de análises concluídas; não inclui falhas. Consultas registradas não provam economia de APIs. Avaliações de usuários não são rótulos de verdade.')
    save_report(directory, 'inventario', report, 'Exportação somente leitura. O CSV para rotulagem não contém a previsão da HÍBRIA. Nenhuma busca ou geração foi executada.')
    print(f'{len(rows)} notícias exportadas. Preencha rotulos.csv antes de calcular acurácia.')
    return 0


def evaluate(rows, labels, include_tuning=False):
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('IDs duplicados no snapshot.')
    seen, groups, joined = set(), set(), []
    exclusions = Counter()
    for label in labels:
        key = label.get('id', '').strip()
        if not key or key not in by_id or key in seen:
            raise ValueError('ID de rótulo vazio, duplicado ou ausente no snapshot: '+key)
        seen.add(key)
        truth = norm(label.get('referencia', ''))
        if not truth:
            exclusions['sem_rotulo'] += 1
            continue
        if truth not in TRUTHS:
            raise ValueError('Referência deve ser verdadeira, falsa ou inconclusiva: '+key)
        if not all(label.get(x, '').strip() for x in ['fonte_referencia', 'justificativa', 'avaliador', 'data_rotulagem']):
            raise ValueError('Rótulo sem fonte, justificativa, avaliador ou data: '+key)
        datetime.fromisoformat(label['data_rotulagem'].strip())
        tuning = norm(label.get('usada_no_ajuste', ''))
        if tuning not in ('sim', 'nao'):
            raise ValueError('Preencha usada_no_ajuste com sim ou nao: '+key)
        if tuning == 'sim' and not include_tuning:
            exclusions['usada_no_ajuste'] += 1
            continue
        row = dict(by_id[key], referencia=truth)
        group = label.get('grupo_noticia', '').strip() or row.get('url_normalizada') or row.get('url') or key
        # Impede contar republicações rotuladas no mesmo grupo como amostras independentes.
        if group in groups:
            raise ValueError('Grupo de notícia duplicado; escolha previamente um representante: '+group)
        groups.add(group)
        prediction = row['classificacao_final']
        if prediction not in LABELS:
            raise ValueError('Classificação desconhecida; defina o mapeamento antes de avaliar: '+prediction)
        joined.append(row)
    exclusions['sem_linha_de_rotulo'] += len(by_id)-len(seen)
    metrics = metrics_for(joined)
    metrics.update(excluidas=dict(exclusions), total_no_snapshot=len(rows),
                   exploratorio_inclui_ajuste=include_tuning,
                   por_versao={v:metrics_for([r for r in joined if r['versao_pipeline']==v])
                               for v in sorted({r['versao_pipeline'] for r in joined})})
    return metrics, joined


def metrics_for(rows):
    matrix = {t:{p:0 for p in LABELS} for t in TRUTHS}
    for r in rows:
        matrix[r['referencia']][r['classificacao_final']] += 1
    binary = [r for r in rows if r['referencia'] in ('verdadeira', 'falsa')]
    decided = [r for r in binary if r['classificacao_final'] in LABELS[:2]]
    # Positivo = notícia falsa detectada como não confiável.
    tp = sum(r['referencia']=='falsa' and r['classificacao_final']=='não confiável' for r in decided)
    tn = sum(r['referencia']=='verdadeira' and r['classificacao_final']=='confiável' for r in decided)
    fp = sum(r['referencia']=='verdadeira' and r['classificacao_final']=='não confiável' for r in decided)
    fn = sum(r['referencia']=='falsa' and r['classificacao_final']=='confiável' for r in decided)
    sensitivity, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    f1_fake, f1_true = ratio(2*tp, 2*tp+fp+fn), ratio(2*tn, 2*tn+fp+fn)
    n = len(binary)
    true_n = sum(matrix['verdadeira'].values())
    fake_n = sum(matrix['falsa'].values())
    return {'matriz':matrix, 'rotuladas':len(rows), 'binarias':n, 'decisoes_binarias':len(decided),
            'abstencoes':n-len(decided), 'cobertura_de_decisao':ratio(len(decided),n),
            'acertos':tp+tn, 'erros':fp+fn,
            'acuracia_seletiva':ratio(tp+tn,len(decided)),
            'ic95_wilson_acuracia_seletiva':wilson(tp+tn,len(decided)),
            'acertos_sobre_todas_binarias':ratio(tp+tn,n),
            'ic95_wilson_acertos_sobre_todas':wilson(tp+tn,n),
            'baseline_classe_majoritaria_todas':ratio(max(true_n,fake_n),n),
            'positivo_definido_como':'falsa / não confiável', 'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'precisao_falsas_decididas':ratio(tp,tp+fp), 'recall_falsas_decididas':sensitivity,
            'especificidade_decididas':specificity,
            'acuracia_balanceada_decididas':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'f1_falsas_decididas':f1_fake,
            'macro_f1_decididas':(f1_fake+f1_true)/2 if f1_fake is not None and f1_true is not None else None,
            'recall_falsas_incluindo_abstencoes':ratio(tp,fake_n),
            'recall_verdadeiras_incluindo_abstencoes':ratio(tn,true_n),
            'resumo': f'{n} notícias com referência binária: {true_n} verdadeiras e {fake_n} falsas. Das verdadeiras, {tn} foram classificadas como confiáveis e {fp} como não confiáveis. Das falsas, {tp} foram classificadas como não confiáveis e {fn} como confiáveis. {n-len(decided)} receberam classificações intermediárias/sem decisão.'}


def calculate(args):
    directory = output_dir(args.output)
    snapshot = json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
    with Path(args.labels).open(encoding='utf-8-sig', newline='') as f:
        labels = list(csv.DictReader(f))
    report, joined = evaluate(snapshot['analises'], labels, args.include_tuning)
    report['proveniencia'] = provenance()
    report['sha256_snapshot'] = digest(args.snapshot)
    report['sha256_rotulos'] = digest(args.labels)
    report['limites'] = ['Acurácia seletiva considera apenas decisões confiável/não confiável.',
        'Resultados parciais ou insuficientes são abstenções, não notícias falsas.',
        'Amostra salva é de conveniência; não prova desempenho em toda a internet.',
        'Intervalos supõem amostras independentes; agrupe republicações e temas muito próximos.',
        'Versões diferentes não são comparação pareada; consulte por_versao.',
        'Score da HÍBRIA não é probabilidade calibrada. Não se calcula Brier/log-loss.']
    save_report(directory, 'acuracia', report, report['resumo'])
    with (directory/'matriz.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['referencia',*LABELS])
        for truth, counts in report['matriz'].items(): writer.writerow([truth,*[counts[p] for p in LABELS]])
    with (directory/'casos.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['id','titulo','url','versao_pipeline','referencia','classificacao_final','score_final']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        writer.writerows({k:csv_safe(v) for k,v in row.items()} for row in joined)
    print(report['resumo'])
    if not report['binarias']:
        print('PENDENTE: nenhuma notícia com referência binária elegível. Acurácia não foi medida.')
        return 2
    return 0


def stress(args):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace as N
    directory = output_dir(args.output)
    report = provenance()
    with isolated():
        from pipeline.output.aggregator import Aggregator
        import faiss
        import numpy as np
        faiss.omp_set_num_threads(1)
        vectors=np.random.default_rng(42).normal(size=(1000,32)).astype('float32')
        faiss.normalize_L2(vectors)
        index=faiss.IndexFlatIP(32);index.add(vectors)
        ev=N(evidence_id='e', evidence_url='https://ref.example/noticia', evidence_layer='web_search',
             source_type='news', trusted_source=True, similarity_final=.95, is_sufficient=True, metadata={})
        def work(i):
            start=time.perf_counter()
            stance=('support','neutral','contradict')[i%3]
            result=N(similarity_scores=[N(claim_id='c',score=.95,has_evidence=True,top_evidence=ev,evidences=[ev])],
                     stance_results=[N(claim_id='c',evidence_id='e',stance=stance)],
                     reputation={'status':'evaluated','score':.95,'note':95},classification={'status':'ok','score':.99})
            actual=Aggregator.aggregate(result)
            expected=('confiável','evidência insuficiente','não confiável')[i%3]
            _, ids=index.search(vectors[i%1000:i%1000+1],5)
            return {'ok':actual['label']==expected and int(ids[0,0])==i%1000,
                    'ms':(time.perf_counter()-start)*1000}
        # Aquecimento fora da medição.
        work(0)
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(work,range(args.iterations)))
        duration=time.perf_counter()-start
    report.update(tipo='carga sintética do agregador + busca FAISS de 1000 vetores de 32 dimensões',
                  iteracoes=args.iterations, workers=args.workers, erros=sum(not r['ok'] for r in results),
                  segundos=duration, operacoes_por_segundo=args.iterations/duration,
                  latencia_ms=percentiles(r['ms'] for r in results),
                  limite='Não mede extração web, API HTTP, Postgres, embeddings nem Qwen. Não representa capacidade de análises completas em produção.')
    save_report(directory,'carga',report,'Carga limitada, em memória, com verificações de resultado e sem APIs. Não altera a instância em produção.')
    return 1 if report['erros'] else 0


def inspect_faiss(args):
    # Nenhum construtor VectorStore nem reparo: somente leitura dos arquivos ativos.
    sys.path.insert(0,str(ENGINE))
    from pipeline.retrieval.vector_storage import active_paths
    import faiss
    directory=output_dir(args.output)
    store=Path(args.store).resolve()
    report=provenance()
    for attempt in range(3):
        first=active_paths(store)
        index=faiss.read_index(str(first[0]))
        metadata=json.loads(first[1].read_text(encoding='utf-8'))
        if first==active_paths(store): break
    else: raise ValueError('O índice mudou durante a leitura. Tente novamente quando estiver ocioso.')
    ok=isinstance(metadata,list) and len(metadata)==index.ntotal and all(isinstance(v,dict) and v.get('faiss_id')==i for i,v in enumerate(metadata))
    report.update(vetores=index.ntotal,metadados=len(metadata),dimensoes=index.d,consistente=ok,
                  caminho=str(first[0]),somente_leitura=True,
                  limite='Confere estrutura e correspondência por índice; não mede qualidade semântica das evidências.')
    save_report(directory,'faiss',report,'Inspeção de integridade do índice existente, sem reparos, embeddings ou buscas web.')
    return 0 if ok else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['testes','exportar','metricas','carga','faiss']:
        p=sub.add_parser(name)
        p.add_argument('--output',required=True,help='Diretório NOVO para preservar execuções anteriores.')
        if name=='exportar':
            p.add_argument('--env-file');p.add_argument('--pipeline-version')
            p.add_argument('--limit',type=int,default=0,help='0: todas; >0: amostra aleatória sem reposição.')
            p.add_argument('--seed',type=int,default=20261001)
        if name=='metricas':
            p.add_argument('--snapshot',required=True);p.add_argument('--labels',required=True)
            p.add_argument('--include-tuning',action='store_true',help='Exploratório: inclui notícias usadas para ajustar o sistema.')
        if name=='carga':
            p.add_argument('--iterations',type=int,default=1000);p.add_argument('--workers',type=int,default=2)
        if name=='faiss':p.add_argument('--store',default=str(ENGINE/'data/vector_store'))
    args=parser.parse_args()
    if args.command=='carga' and not (1<=args.iterations<=10000 and 1<=args.workers<=4):
        parser.error('Use 1..10000 iterações e 1..4 workers.')
    if args.command=='exportar' and args.limit<0:parser.error('--limit não pode ser negativo.')
    try:
        return {'testes':run_tests,'exportar':export_database,'metricas':calculate,'carga':stress,'faiss':inspect_faiss}[args.command](args)
    except Exception as exc:
        # Não imprime DSNs, senhas, conexão ou dados de ambiente.
        print(f'Falha ({type(exc).__name__}).',file=sys.stderr)
        if isinstance(exc,(ValueError,FileExistsError,FileNotFoundError,ImportError)):
            print(str(exc),file=sys.stderr)
        else: print('Verifique dependências, permissões e disponibilidade do banco/arquivos. Nenhum reparo foi executado.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
