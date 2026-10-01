# Validação da HÍBRIA sem novas consultas de pesquisa

Este pacote adiciona um comando em `ai-engine/scripts/avaliar_hibria_offline.py`.
Não altera a extensão, o prompt, os modelos, a classificação ou o serviço da AWS.
A suíte usa a implementação real dos componentes, com dados artificiais e mocks
nas integrações. Nenhum teste chama a pipeline completa com provedores ativos.

## O que cada execução mede

| Comando | O que mede | O que não comprova |
|---|---|---|
| `testes` | Contratos, segurança HTTP/SSRF, flags, cotas, cache, cancelamento, fila, persistência simulada, FAISS temporário, cálculo e validação da explicação | Acurácia factual das notícias; qualidade de texto de uma execução real do Qwen |
| `carga` | Agregação real e busca FAISS sintética em paralelo, erros, operações/s e latências | Análises completas/s; capacidade do Qwen; desempenho HTTP/Postgres em produção |
| `faiss` | Leitura e integridade estrutural do par índice/metadados existente | Correção factual e qualidade semântica dos documentos |
| `exportar` | Resultados já salvos, tempos históricos, distribuição de classes e avaliações de usuários; cria ficha cega para rotulação | Acurácia sem rótulos; falhas de análises que nunca foram salvas |
| `metricas` | Confronto entre classificação salva e referência independente | Desempenho da versão atual em registros produzidos por versões anteriores |

As rotinas não executam pesquisas, Ollama, download de modelos, migrações nem
reparos. Os testes bloqueiam conexões de rede, processos externos e conexões reais
ao banco, neutralizam dotenv e usam diretórios temporários. Não execute diretamente
`unittest discover` com o ambiente de produção: use o comando `testes` com isolamento.
A exportação abre uma transação PostgreSQL **somente leitura**, em REPEATABLE READ.
O diagnóstico FAISS deve ser executado quando não houver indexação em andamento.

## Testes incluídos

Os três arquivos existentes na main são fornecidos sem alteração:
`test_regressions.py`, `test_rag_memory.py`, `test_reputation.py` (20 casos).
Os testes de revisão, antes externos, passam a ficar em:

- `test_validacao_controles.py`: autorização, cancelamento, limites, pré-requisitos e direção da nota.
- `test_validacao_integridade.py`: persistência, filtro das evidências e falhas de componentes.
- `test_validacao_seguranca.py`: SSRF, redirecionamentos, segredos, cache, fila, feedback e gravação atômica FAISS.
- `test_validacao_metricas.py`: contas, denominadores, abstenções, deduplicação, referência independente e exportação somente leitura.

O relatório lista cada caso, duração, falha ou erro. Teste ignorado não conta como
aprovação completa. Erros simulados no console podem ser esperados: confira o resumo
final e `testes.log`. A suíte não depende de credenciais reais.

## AWS: primeiros comandos

Depois de instalar os arquivos do ZIP no repositório:

```bash
cd ~/HIBRIA/ai-engine
source ~/HIBRIA/.venv/bin/activate
pasta="$HOME/hibria-validacao-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$pasta"
printf '%s\n' "$pasta" > "$HOME/hibria-validacao-ultima.txt"
python scripts/avaliar_hibria_offline.py testes --output "$pasta/testes"
python scripts/avaliar_hibria_offline.py carga --iterations 2000 --workers 2 --output "$pasta/carga"
python scripts/avaliar_hibria_offline.py faiss --output "$pasta/faiss"
```

A carga é limitada a 10.000 operações e quatro workers, padrão dois. São consultas
em um índice temporário de 1.000 vetores de 32 dimensões. Não cria jobs de produção.
Não é necessário reiniciar o serviço ou recriar o ZIP da extensão.

As dependências são as já usadas pelo backend; `httpx` está no requirements-aws.lock.
Se faltar alguma, não reinstale Torch/modelos. Identifique a dependência no log e use
a versão indicada no lock do projeto. Não é necessária uma API paga para instalação.

## Listar e preparar notícias que já foram analisadas

```bash
pasta="$(cat "$HOME/hibria-validacao-ultima.txt")"
sudo /home/ubuntu/HIBRIA/.venv/bin/python \
  /home/ubuntu/HIBRIA/ai-engine/scripts/avaliar_hibria_offline.py exportar \
  --env-file /home/ubuntu/.config/hibria/hibria.env \
  --output "$pasta/amostra"
sudo chown -R "$(id -u):$(id -g)" "$pasta/amostra"
```

`inventario.html` lista versões e quantidades. `analises.json` é o snapshot das
previsões; `rotulos.csv` contém somente identificação, título, URL, conteúdo e campos
vazios para referência. O avaliador humano deve receber **só o CSV**.
Não mostre o resultado do modelo antes da rotulação.

Por padrão, exporta todas as notícias salvas e mantém somente a revisão mais recente
por URL ou conteúdo idêntico. Para uma amostra aleatória reprodutível, acrescente
`--limit 100 --seed 20261001`. Para estudar apenas uma versão, acrescente
`--pipeline-version 'valor exato do inventario'` e use outro diretório de saída.
O filtro é aplicado antes da deduplicação. Registros antigos não são reanalisados.

Não crie milhares de notícias artificiais e as apresente como notícias reais.
Sem novas APIs, a avaliação completa fica limitada aos resultados e referências já
coletados. Notícias ainda não analisadas precisam de evidências pré-coletadas ou de
uma coleta adicional. Executar apenas BERTimbau em um corpus não avalia toda a HÍBRIA.
Fake.Br usado para treinamento ou povoamento do RAG pode contaminar a avaliação;
não use o mesmo material como teste independente de generalização.

## Preencher a referência

Use UTF-8 e mantenha os nomes das colunas do CSV. Excel/LibreOffice pode mudar o
separador; ao salvar, selecione CSV separado por **vírgula**. Não edite IDs.
As quebras de linha do conteúdo são escapadas pelo formato CSV.

| Campo | Como preencher |
|---|---|
| `referencia` | `verdadeira`, `falsa` ou `inconclusiva` |
| `fonte_referencia` | Documento primário, verificação independente ou referência arquivada que sustente o rótulo |
| `justificativa` | Motivo do rótulo e qual informação central foi verificada |
| `avaliador` | Código do avaliador, por exemplo A1, sem nome pessoal |
| `data_rotulagem` | Data ISO, por exemplo `2026-10-01` |
| `grupo_noticia` | Mesmo código para republicações/matérias sobre o mesmo fato; escolha previamente um representante por grupo |
| `usada_no_ajuste` | `sim` para notícias usadas para ajustar prompt, regras ou modelo; `nao` para as demais |

Defina antes: verdadeira = informações factuais centrais sustentadas; falsa = erro
factual central demonstrado; inconclusiva = evidência insuficiente, conteúdo misto
sem um rótulo global defensável ou falta de referência adequada. Não marque verdadeira
só porque veio de um veículo conhecido. A nota da HÍBRIA, o texto do Qwen e as estrelas
de usuários não podem ser a fonte do rótulo.

As notícias usadas repetidamente no desenvolvimento (por exemplo os exemplos de
pesquisas eleitorais e entretenimento do laboratório) são casos de ajuste: marque
`sim`. São excluídas das métricas principais. `--include-tuning` gera avaliação
explicitamente exploratória, não um conjunto independente.

Para o TCC, prefira dois avaliadores independentes e resolva divergências antes de
produzir o CSV final. Guarde as duas fichas originais e as decisões de consenso.
Inclua verdadeiras e falsas, temas e fontes variados; fixe a seleção antes de olhar
os acertos. Uma amostra pequena ou só de notícias verdadeiras não comprova detecção
de desinformação. Arquive as referências para considerar data e mudanças da notícia.
Não existe um número mágico de notícias que garanta viabilidade.

## Calcular resultados sem gastar API

```bash
cd ~/HIBRIA/ai-engine
source ~/HIBRIA/.venv/bin/activate
pasta="$(cat "$HOME/hibria-validacao-ultima.txt")"
python scripts/avaliar_hibria_offline.py metricas \
  --snapshot "$pasta/amostra/analises.json" \
  --labels "$pasta/amostra/rotulos.csv" \
  --output "$pasta/metricas-$(date +%Y%m%d-%H%M%S)"
```

Gera HTML, JSON, matriz CSV e lista dos casos. Sem rótulos binários elegíveis, retorna
código 2 e deixa as métricas como nulas: **não foi medida acurácia**. Campos inválidos,
IDs duplicados e grupos repetidos interrompem o cálculo. Corrija o CSV e use um novo
diretório. Os arquivos anteriores não são sobrescritos.

O resumo informa: “X notícias com referência binária: Y verdadeiras e Z falsas.
Das verdadeiras, ... foram classificadas como confiáveis ...”. A matriz mantém as
cinco classificações do sistema, inclusive resultados intermediários.

- Decisão binária: `confiável` → previsão verdadeira; `não confiável` → previsão falsa.
- Abstenção: `parcialmente confiável`, `evidência insuficiente` e `não verificado`.
- Acurácia seletiva = acertos / decisões binárias. Apresente sempre junto da cobertura.
- Cobertura de decisão = decisões binárias / notícias com referência binária.
  Esta cobertura é da avaliação; não é o `coverage_score` interno do pipeline.
- Acertos sobre todas = acertos / todas as notícias com referência verdadeira/falsa,
  mantendo as abstenções no denominador. Não chama abstenção de notícia falsa.
- Positivo = notícia falsa. Precisão, recall, especificidade, F1 e acurácia balanceada
  das decisões são identificados como seletivos. Há também recall com abstenções.
- O baseline majoritário é calculado sobre toda a amostra binária; compare com acertos
  sobre todas, não diretamente com a acurácia seletiva de um subconjunto.
- IC95% Wilson acompanha as proporções de acerto; é descritivo e pressupõe observações
  independentes. Casos relacionados podem tornar esse intervalo otimista.
- Valores sem denominador são `null`, não 0 ou 100%. Versões são discriminadas.
- A nota de confiabilidade não é probabilidade calibrada; não se calcula Brier/log-loss.

Referências metodológicas: documentação oficial de métricas de classificação do
scikit-learn, https://scikit-learn.org/stable/modules/model_evaluation.html ;
intervalos binomiais Wilson: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm .
O cálculo usa a biblioteca padrão, sem instalar scikit-learn.

## Evidências adicionais para a escrita

Use conjuntamente: resultado dos testes, matriz e denominadores, versões do código,
intervalos, latência histórica, limites da amostra e feedback de usuários. Tempos no
banco representam apenas execuções concluídas; não estime taxa de falhas a partir deles.
A média das estrelas mede percepção dos participantes e é autorselecionada.

Faça também uma observação manual, opcional, com uma notícia já salva:

1. Reabrir a extensão: mantém o resultado e a identificação da notícia.
2. Nova análise da mesma notícia: retorna o cache da versão vigente, quando elegível.
3. Trocar de aba: preserva a análise escolhida até clicar em Nova Análise.
4. Teclado, foco visível, modo claro/escuro e estrelas de meia em meia.
5. Votar uma vez: confirmar associação à análise e impedimento de voto repetido.

Os passos que enviam feedback alteram o banco; não fazem parte do runner offline.
Não prometa custo zero ao testar manualmente uma notícia nova ou cache expirado.
Testes de qualidade do Qwen com contextos salvos podem rodar localmente, mas ainda
consomem CPU/RAM e tempo. Esta suíte não altera ou retesta a redação com o modelo.

## Reprodutibilidade e integridade

Mantenha snapshot, CSV rotulado, relatório, hashes SHA-256, versão da pipeline e
commit juntos. O script não muda classificação nem calcula uma nota favorável para
passar na avaliação. Separe desenvolvimento e avaliação final; se ajustar o sistema
com base em erros, avalie depois em outra amostra independente.
Não faça commit dos resultados com conteúdo de notícias, rótulos pessoais ou .env.
O pacote não consulta o banco da sua AWS a partir do chat: os resultados reais serão
produzidos quando você executar os comandos no servidor.
