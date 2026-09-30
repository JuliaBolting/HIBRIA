# Revisão final: integridade e operação

Base: main d826a0c. A extensão mantém as quatro etapas, as mensagens de espera,
a retomada do popup e o visual anterior. A versão do pacote é 1.2.2.

## Mudanças funcionais

- A autorização usa o caminho ASGI, independente do cabeçalho Host. Hosts
  malformados são recusados. FastAPI e Starlette foram atualizados em conjunto.
- Os requisitos de execução substituem o antigo freeze que incluía ferramentas
  sem uso no projeto. `requirements-aws.lock` fixa o ambiente validado, com Torch
  CPU. Instale em ambiente novo; instalar por cima não remove pacotes antigos.
- O cancelamento interrompe novas consultas entre camadas, pontos, reservas de
  cota e chamadas HTTP. Uma chamada que já foi enviada pode consumir sua cota e
  terminar até o timeout; não há promessa de recuperar créditos já utilizados.
- O FAISS avança sempre ao dividir um texto em blocos, inclusive quando há uma
  palavra maior que o limite. O último bloco encerra o laço.
- Todas as camadas descartam a própria URL analisada. Referências elegíveis têm
  prioridade antes do corte dos resultados; a deduplicação conserva a melhor
  representação de um mesmo documento.
- A reputação dinâmica também alcança domínios das referências, com no máximo
  **2 novas avaliações por análise**, compartilhadas entre os pontos. O limite é
  `HIBRIA_REFERENCE_REPUTATION_BUDGET` (0 desativa novas avaliações, máximo 10).
  Avaliações existentes e válidas são reutilizadas. Fontes pendentes continuam
  identificadas como não avaliadas; não viram confiáveis automaticamente.
  As consultas de reputação usam as mesmas cotas dos provedores.
- Provedor desligado, sem resultados ou pulado não é descrito como falha de busca.
  O texto alternativo não inventa um ponto sem apoio quando todos os pontos
  registrados receberam apoio. O prompt continua genérico, sem notícias de teste.
- Uma análise só termina com sucesso após gravar o resultado principal. O banco
  é verificado antes da extração e das consultas. Erros dos jobs ficam nos logs
  com classe e localização, sem argumentos contendo credenciais.

## Cálculo e histórico

Os pesos configurados dos componentes foram preservados. O componente factual
passou a considerar a direção das comparações: similaridade de apoio soma,
contradição subtrai, neutralidade ou insuficiência não soma. São usadas somente
referências elegíveis que passaram pela etapa de comparação; médias por ponto
evitam dar mais peso apenas por haver mais documentos sobre o mesmo ponto.
A média é multiplicada pela cobertura. Reputação e BERTimbau mantêm seus pesos
auxiliares. O índice final permanece entre 0 e 100 e não é uma probabilidade de
verdade. A ausência de evidência não recebe o rótulo "não confiável" por si só.

A identificação de cache inclui `:directional-2`, mesmo com um valor antigo de
HIBRIA_PIPELINE_VERSION no env. Portanto a primeira consulta de uma URL antiga
será recalculada e pode consumir APIs; as seguintes reutilizam o novo resultado.
As análises anteriores ficam no banco e não são apagadas nem recalculadas em lote.

A migração 005 remove a sobrescrita de resultados por URL/conteúdo/versão. Uma
nova execução cria um ID; uma consulta ao cache mantém o ID. Quando só a redação
muda, é criada uma revisão com novo ID, ligada à anterior, com cópia das relações
RAG e a mesma data da análise original para não renovar artificialmente o cache.
Os votos antigos continuam associados ao resultado que foi avaliado.

As avaliações aceitam 0,5 a 5 estrelas: abaixo de 3 é negativa, de 3 a 3,5 é
neutra, a partir de 4 é positiva. As categorias antigas de notas inteiras são
preservadas. As médias e resumos são views atualizadas a cada leitura.

## Implantação

Aplicar a migração 005 **antes** de iniciar esta versão. O instalador do pacote
faz backup do banco, ambiente e commit anterior e prepara um ambiente Python
separado. O backend antigo não é compatível com a remoção da chave única da
migração 005; voltar somente o Git não basta. Consulte as instruções de retorno
no pacote antes de reverter uma instalação que já tenha recebido novos dados.

`ai-engine/main.py` exige uma URL por argumento e escreve em data/runtime por
padrão. Ele não executa uma notícia de exemplo ao ser importado.

O submódulo opcional Fake.Br agora possui .gitmodules. Não é baixado pelo
instalador de produção e permanece excluído da recuperação de evidências.

Os testes novos desta revisão ficam no pacote de validação, fora da main.
