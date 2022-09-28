# Scientific Recommender

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
* [semanticscholar](https://pypi.org/project/semanticscholar/)

### Funcionamento

O código funciona através de um Web Scraping no site do [DBLP](https://dblp.org/), que consulta a listagem de artigos publicados nas conferências mencionadas entre os anos de 2018 e 2022. A partir dos dados coletados, e com as informações complementadas pela api do Semantic Scholar e o site da [DL ACM](https://dl.acm.org/), o dataset é gerado e salvo num arquivo [CSV](/Dataset/articles-2022-09-19.csv).

### Como rodar

1. Instale as dependências Python que o código possui
2. > python3 getArticleData.py

## Artigo

O artigo pode se encontrar de forma [`online`](https://www.overleaf.com/8891746977wpkbvdfbzjdn) ou na pasta [Artigo](/Artigo/Scientific_Recommender.pdf).

## Cite As

Ingrid Pacheco, Eduardo Prata & Renan Parreira. (2022, September 27). ingridpacheco/Scientific-Recommender: Repository of Scientific Recommender. 

** Scientific Recommender @ copyright, Ingrid Pacheco, Eduardo Prata e Renan Parreira, 2022**
