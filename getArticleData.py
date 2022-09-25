# -*- coding: utf-8 -*-
"""

# **MAI712** - Fundamentos em Ciência de Dados
___
#### **Professores:** Sergio Serra e Jorge Zavaleta
___
#### **Equipe:** Ingrid Pacheco, Eduardo Prata, Renan Parreira
___
### **OBJETIVO:**
Script de criação de um dataset com os artigos publicados em alguns eventos para o projeto final da matéria MAI712

#### **Requisito:**
É necessário que tenhamos instalado o Beautiful Soup, antes mesmo de iniciarmos nosso _** de Web Scraping**_ para construção do nosso _DataSet_ de artigos.

#### **Imports e Bibliotecas**

Aqui estaremos declarando as bibliotecas e módulos necessários para nosso scrtipt
"""

# Importar módulo de Requests e de Expressões Regulares
from multiprocessing import AuthenticationError
import requests, re, random, json
# Import para tratamento do arquivo e da versão do sistema
from os import system, name, remove
from os.path import isfile
# Import do beautifulSoup para tratamento dos dados retornados da pagina Web
from bs4 import BeautifulSoup
from datetime import date
import csv
import time

import urllib, urllib.request

import semanticscholar as sch

def findArticle(title):
  filtro = title.replace(' ','+')
  link = 'https://api.semanticscholar.org/graph/v1/paper/search?query='
  print('Não tem DOI')
  retorno = requests.get( link + filtro, headers = cabecalho)
  if retorno.status_code == 429:
      time.sleep(300)
      print('Esperando um pouco')
      retorno = requests.get( link + filtro, headers = cabecalho)
  sp_artigo = BeautifulSoup(retorno.text, 'html.parser')
  artigo_json = json.loads(sp_artigo.text)
  if len(artigo_json["data"]) > 0:
    str_paperId = artigo_json["data"][0]["paperId"]
    return str_paperId
  return None

def get_complementary_data(qty_article, search_id):

  if qty_article % 99 == 0 and qty_article != 0:
    print("Atingiu +99")
    time.sleep(300)
  retorno = sch.paper(search_id, timeout=8)

  while retorno is None:
    print("Deu erro")
    time.sleep(300)
    retorno = sch.paper(search_id, timeout=8)
  
  if 'paperId' in retorno:
    paperId = retorno['paperId']
  else:
    paperId = ''
  
  if 'doi' in retorno:
    doi = retorno['doi']
  else:
    doi = ''

  if 'authors' in retorno:
    authors_list = retorno['authors']
  else:
    authors_list = []

  if 'venue' in retorno:
    publisher = retorno['venue']
  else:
    publisher = ''
  
  if 'topics' in retorno:
    topics_list = retorno['topics']
  else:
    topics_list = []

  if 'fieldsOfStudy' in retorno:
    fields_of_study_list = retorno['fieldsOfStudy']
  else:
    fields_of_study_list = []

  authors = ''
  for author in authors_list: authors = authors + author['name'] + ','
  authors = authors[0:len(authors)-1]

  topics = ''
  for topic in topics_list:
    topic = topic if type(topic) is not dict else topic['topic']
    topics = topics + topic + ','
  topics = topics[0:len(topics)-1]

  fields_of_study = ''
  if fields_of_study_list is not None:
    for field in fields_of_study_list: fields_of_study = fields_of_study + field + ','
    fields_of_study = fields_of_study[0:len(fields_of_study)-1]

  article_data = [paperId,doi,authors,publisher,topics,fields_of_study]
  return article_data

"""#### **Primeiro Etapa:** Declaração das variáveis necessárias para nosso WebScraping

"""

# Pegando aleatoriedade de browser para não ser identificado o Robo.
UAS = ("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1", 
       "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0",
       "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
       )

ua = UAS[random.randrange(len(UAS))]

# Variáveis do sistema
cabecalho = {'user-agent': ua}
header = ['title', 'paperId', 'doi', 'authors', 'publisher', 'topics', 'fields_of_study','year']
today = date.today()
nome_arquivo = 'articles-' + str(today) + '.csv'
qty_article = 0

alvo = "https://dblp.org/"

conferences = ["SOSP - ACM Symposium on Operating Systems Principles",
"OSDI - Operating Systems Design and Implementation",
"NDSS - Network and Distributed System Security Symposium",
"MobiHoc - Mobile Ad Hoc Networking and Computing",
"SIGCOMM - ACM SIGCOMM Conference",
"SenSys - Conference On Embedded Networked Sensor Systems",
"MOBICOM - Mobile Computing and Networking",
"CIDR - Conference on Innovative Data Systems Research",
"USENIX Security Symposium",
"EUROCRYPT - Theory and Application of Cryptographic Techniques"]

"""#### **Segunda Etapa:** Request no alvo e `html.parser` e filtra penas _tags_ analisadas como importantes


"""

with open(nome_arquivo,'w',newline='',encoding="utf-8") as f:

  # create the csv writer
  writer = csv.writer(f,delimiter=";")

  # write header to csv file
  writer.writerow(header)

# Busca o HTML a ser explorado
  for conf in conferences:
    conf = conf.split('-')[0]
    print('Conf: ' + conf)
    alvo = "https://dblp.org/search?q=" + conf
    resposta = requests.get(alvo, headers = cabecalho)

    # Formatando o retorno
    sopa = BeautifulSoup(resposta.text, 'html.parser')

    infos = sopa.find('div', {'id':'completesearch-venues'})
    
    venue = infos.find('ul', {'class':'result-list'}).find('a')

    print('Href: ' + venue.get('href'))
    venue_href = venue.get('href')
    resposta = requests.get(venue_href, headers = cabecalho)
    sopa = BeautifulSoup(resposta.text, 'html.parser')
    for conf_edition in sopa.find_all('ul', {'class':'publ-list'}):
      conf_title = conf_edition.find('span',{'itemprop':'name'}).get_text().lower()

      # Desconsidera os workshops
      if 'workshop' in conf_title:
        continue

      published_year = conf_edition.find('span',{'itemprop':'datePublished'}).get_text()
      print('Ano: ' + published_year)

      # So pega os artigos ate 2018
      if int(published_year) < 2018:
        break
      content = conf_edition.find('a',{'class':'toc-link'}).get('href')
      print(content)
      resposta = requests.get(content, headers = cabecalho)
      sopa = BeautifulSoup(resposta.text, 'html.parser')
      for article in sopa.find_all('li', {'class':'entry inproceedings'}):
        header = article.find_all_previous('h2')[0].get('id').lower()
        
        # Desconsidera os keynotes
        if 'keynote' in header:
          continue
        title = article.find('span', {'class':'title'}).get_text()
        print(title)
        year = article.find('meta', {'itemprop':'datePublished'})['content']
        print(year)
        genre = article.find('meta', {'property':'genre'})['content']
        print(genre)
        doi_url = article.find('li', {'class': 'drop-down'}).find('a').get('href')
        doi = ''
        article_data = []
        if 'https://doi.org/' not in doi_url:
          paperId = findArticle(title)
          if paperId is None:
            print('paperId is None')
            authors = ''
            for author in article.find_all('span',{'itemprop':'author'}):
              authors = authors + author.find('span',{'itemprop':'name'}).get('title') + ','
            authors = authors[0:len(authors)-1]
            article_data = ['','',authors,conf,'',genre]
          else:
            print('paperId: ' + paperId)
            article_data = get_complementary_data(qty_article, paperId)
            print('articledata1: ', article_data)
            doi = article_data[1]
            print('doi: ', doi)

        else:
          doi = doi_url.split('https://doi.org/')[1]
          print(doi)
          article_data = get_complementary_data(qty_article, doi)
          print('articledata2: ', article_data)

        authors = ''
        keywords = ''

        if doi != '':
          resposta = requests.get(doi_url, headers = cabecalho)
          sopa = BeautifulSoup(resposta.text, 'html.parser')
          # print(sopa.prettify())
          author_list = sopa.find('ul',{'ariaa-label':'authors'})
          if author_list is not None:
            for author in author_list.find_all('a',{'class':'author-name'}):
              authors = authors + author.get('title') + ','
            authors = authors[0:len(authors)-1]
          print(authors)

          kw_list = sopa.find('ol',{'class':'rlist organizational-chart'})
          if kw_list is not None:
            for kw in kw_list.find_all('a'):
              keywords = keywords + kw.get_text() + ','
            keywords = keywords[0:len(keywords)-1]
          print(keywords)
        
        if article_data is not None:
          article_data.insert(0,title)
          article_data.append(year)
          article_data[3] = authors if authors != '' else article_data[3]
          article_data[4] = conf
          if article_data[5] == '' and keywords != '':
            article_data[5] = keywords
          elif article_data[5] != '' and keywords != '':
            article_data[5] = article_data[5] + ', ' + keywords
          # [title,paperId,doi,authors,publisher,topics,fields_of_study,year]
          if article_data[6] == '':
            article_data[6] = genre
          elif genre.lower() not in article_data[6].lower():
            article_data[6] = article_data[6] + ', ' + genre
        else:
          article_data = [title,'',doi,authors,conf,keywords,genre,year]

        writer.writerow(article_data)
        qty_article += 1
print('Quantidade: ' + qty_article)