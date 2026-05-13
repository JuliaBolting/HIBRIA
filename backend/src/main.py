from pipeline.preprocessing.extractor import TextExtractor

data = TextExtractor.extract(
    "https://g1.globo.com/sp/bauru-marilia/noticia/2026/05/11/policia-encontra-cerveja-e-cooler-em-carro-de-motorista-suspeito-de-provocar-acidente-que-matou-quatro-jovens-no-dia-das-maes.ghtml"
)

print(data)