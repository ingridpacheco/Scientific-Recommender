# Scientific Recommender [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7145542.svg)](https://doi.org/10.5281/zenodo.7145542)

**Este projeto está licenciado nos termos da licença MIT.**

Projeto final da disciplina de Fundamentos de Ciência de Dados do mestrado do PPGI.

**Autores**: Ingrid Pacheco, Eduardo Prata e Renan Parreira

**Professore**: Sérgio Serra e Jorge Zavatela

## Objetivo

Criação de um sistema de recomendação de conferências e jornals. Entretanto, é necessária uma análise para entender o cenário atual das conferências em relação aos tópicos que mais são publicados e sua evolução histórica. Este trabalho, portanto, se propõe a fazer uma análise histórica de tópicos publicados em 10 conferências entre os anos de 2018 e 2022, a fim de ter uma melhor visualização dos seus padrões para auxiliar na implementação futura do sistema de recomendação.

## Arquivos

O projeto é composto por:

* [Código de criação do dataset](/getArticleData.py)
* [Código de análise do dataset](/articleAnalysis.py)
* [Artigo](/Artigo/Scientific_Recommender.pdf)
* [Dataset gerado](/Dataset/articles-2022-09-19.csv)
* [Imagens geradas](/Imagens/)
* [Notebooks](/Notebooks/)
* [Proveniências geradas](/Proveniencia/)
* [Apresentação](/Apresentacao/)
* [Licença](/LICENSE)
* [README do projeto](/README.md)

## Geração do Dataset

Com o objetivo de realizar as análises nos artigos publicados em diferentes conferências, foi necessária a criação de um dataset único.

### Conferências

As conferências utilizadas para a criação do Dataset foram extraídas de uma lista provida pela [Universidade Cornell](https://www.cs.cornell.edu/andru/csconf.html).

As conferências, portanto, escolhidas foram:

* "SOSP - ACM Symposium on Operating Systems Principles",
* "OSDI - Operating Systems Design and Implementation",
* "NDSS - Network and Distributed System Security Symposium",
* "MobiHoc - Mobile Ad Hoc Networking and Computing",
* "SIGCOMM - ACM SIGCOMM Conference",
* "SenSys - Conference On Embedded Networked Sensor Systems",
* "MOBICOM - Mobile Computing and Networking",
* "CIDR - Conference on Innovative Data Systems Research",
* "USENIX Security Symposium",
* "EUROCRYPT - Theory and Application of Cryptographic Techniques"

### Bibliotecas

As bibliotecas utilizadas no código de geração do dataset são:

* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [prov](https://pypi.org/project/prov/)
* [semanticScholar - versão 0.3.0](https://pypi.org/project/semanticscholar/)
* [requests](https://pypi.org/project/requests/)
* [IPython](https://ipython.org/install.html)

### Funcionamento

O código funciona através de um Web Scraping no site do [DBLP](https://dblp.org/), que consulta a listagem de artigos publicados nas conferências mencionadas acima entre os anos de 2018 e 2022. A partir dos dados coletados, e com as informações complementadas pela api do Semantic Scholar e o site da [DL ACM](https://dl.acm.org/), o dataset é gerado e salvo num arquivo [CSV](/Dataset/articles-2022-09-19.csv).

### Como rodar

1. Instale as dependências Python que o código possui
2. > python3 getArticleData.py

### Notebook

Também está disponibilizado o [Notebook](/Notebooks/getArticleData.ipynb) do código para reprodução do mesmo.

### Dataset disponibilizado

O dataset está disponibilizado tanto no [Kaggle](https://www.kaggle.com/datasets/ingridpacheco/published-articles?select=articles-2022-09-19.csv) quanto no [Google Drive](https://drive.google.com/uc?id=1kWTbqT4QXZ2cVIP5dIHbgVI6ipRc3ba1&authuser=0&export=download), inclusive utilizado no código de análise do Dataset para reprodução do resultado.

## Análise do Dataset

Para realizar as análises sobre o dataset de publicações gerado, foi criado um [script](/articleAnalysis.py) que analisa as conferências e tópicos e seus históricos, a fim de responder 9 perguntas:

* Quais são os 6 tópicos mais publicados nos eventos?
* Quais são os 6 tópicos mais publicados por ano (de 2018 a 2022) nestes eventos?
* Para cada tópico, quais as conferências que mais os publicam?
* Como foram as evoluções para cada tópico das conferências ao longo dos anos?
* Para cada tópico, em que categoria as conferências entram em relação a publicação dos mesmos?
* Quais as conferências que mais publicaram?
* Como foram as evoluções ao longo dos anos para as conferências?
* Para cada conferência, qual o tópico que mais aparece?
* Como foram as evoluções dos tópicos publicados por cada conferência ao longo dos anos?

### Análises

As análises realizadas se mantém nas perguntas mencionadas acima, e para isso se utilizam de estruturas de dicionário criadas para auxiliar o processo. Além disso, também é realizado o processo de clusterização para entender como as conferências se comportam em termos de publicação para cada um dos tópicos.

### Bibliotecas

As bibliotecas utilizadas no código de análise do dataset são:

* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [sklearn](https://scikit-learn.org/)
* [bokeh - versão 2.4.3](http://bokeh.org/)
* [plotly.express](https://pypi.org/project/plotly-express/)
* [seaborn](https://seaborn.pydata.org/)
* [prov](https://pypi.org/project/prov/)
* [IPython](https://ipython.org/install.html)

### Como rodar

1. Instale as dependências Python que o código possui
2. > python3 articleAnalysis.py

### Notebook

Também está disponibilizado o [Notebook](/Notebooks/articleAnalysis.ipynb) do código para reprodução do mesmo.

### Resultado

Todas as imagens geradas das análises executadas estão na pasta [Imagens](/Imagens/).

## Artigo

O artigo se encontra na pasta [Artigo](/Artigo/Scientific_Recommender.pdf).

## Proveniência

Com o auxília da biblioteca `prov` foi possível gerar a proveniência de ambos os códigos ([geração de dataset](/getArticleData.py) e [análise do mesmo](/articleAnalysis.py)). Todas as imagens geradas da proveniência se encontram na pasta [Proveniencia](/Proveniencia/).

## Cite As

Ingrid Pacheco, & Eduardo Prata. (2022). ingridpacheco/Scientific-Recommender: v1.0 (1.0). Zenodo. https://doi.org/10.5281/zenodo.7145542

**Scientific Recommender @ copyright, Ingrid Pacheco e Eduardo Prata, 2022**
