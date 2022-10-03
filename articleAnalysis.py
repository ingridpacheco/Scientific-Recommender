"""
# **MAI712** - Fundamentos em Ciência de Dados
___
#### **Professores:** Sergio Serra e Jorge Zavaleta
___
#### **Equipe:** Ingrid Pacheco, Eduardo Prata
___
### **OBJETIVO:**
Script de análise de artigos publicados em alguns eventos para o projeto final da matéria MAI712

#### **Imports e Bibliotecas**

Aqui estaremos declarando as bibliotecas e módulos necessários para nosso script:

* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [sklearn](https://scikit-learn.org/)
* [bokeh - versão 2.4.3](http://bokeh.org/)
* [plotly.express](https://pypi.org/project/plotly-express/)
* [seaborn](https://seaborn.pydata.org/)
* [prov](https://pypi.org/project/prov/)
* [IPython](https://ipython.org/install.html)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import os
import urllib.request

from bokeh.palettes import Category10  
from bokeh.plotting import figure, show            # paleta de cores
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot

import plotly.express as px
import seaborn as sns

import datetime
from prov.model import ProvDocument
from prov.dot import prov_to_dot

from IPython.display import Image

d1 = ProvDocument()

dagnts = {}
dentities = {}
dactivities = {}

def create_dicts(article_data):
    topics_publisher_year = {}
    topics_year_publisher = {}
    topics_qty = {}
    articles_by_publisher = {}
    publisher_qty = {}
    topics_qty_year = {}
    publisher_qty_year = {}
    year_topics = {}

    for i,article in enumerate(article_data.values):
        print('qty: ' + str(i))
        print('artigo: ' + article[0])
        print('topics: ')
        topics = article[5]
        print(topics)
        publisher = article[4]
        year = article[7]
        if not pd.isna(publisher):
            if publisher not in publisher_qty_year:
                    publisher_qty_year[publisher] = {}
                    publisher_qty_year[publisher][year] = 1
            elif year not in publisher_qty_year[publisher]:
                    publisher_qty_year[publisher][year] = 1
            else:
                    publisher_qty_year[publisher][year] += 1

        if not pd.isna(publisher) and not pd.isna(topics):
            for topic in topics.split(','):
                topic = topic.upper()

                if year not in year_topics:
                    year_topics[year] = {}
                    year_topics[year][topic] = 1
                elif topic in year_topics[year]:
                    year_topics[year][topic] += 1
                else:
                    year_topics[year][topic] = 1

                if topic not in topics_publisher_year:
                    topics_publisher_year[topic] = {}
                    topics_publisher_year[topic][publisher] = {}
                    topics_publisher_year[topic][publisher][year] = 1
                elif publisher in topics_publisher_year[topic]:
                    if year not in topics_publisher_year[topic][publisher]:
                        topics_publisher_year[topic][publisher][year] = 1
                    else:
                        topics_publisher_year[topic][publisher][year] += 1
                else:
                    topics_publisher_year[topic][publisher] = {}
                    topics_publisher_year[topic][publisher][year] = 1
                
                if topic not in topics_year_publisher:
                    topics_year_publisher[topic] = {}
                    topics_year_publisher[topic][year] = {}
                    topics_year_publisher[topic][year][publisher] = 1
                elif year in topics_year_publisher[topic]:
                    if publisher not in topics_year_publisher[topic][year]:
                        topics_year_publisher[topic][year][publisher] = 1
                    else:
                        topics_year_publisher[topic][year][publisher] += 1
                else:
                    topics_year_publisher[topic][year] = {}
                    topics_year_publisher[topic][year][publisher] = 1

                if topic not in topics_qty:
                    topics_qty[topic] = 1
                else:
                    topics_qty[topic] += 1

                if topic not in topics_qty_year:
                    topics_qty_year[topic] = {}
                    topics_qty_year[topic][year] = 1
                else:
                    if year not in topics_qty_year[topic]:
                        topics_qty_year[topic][year] = 1
                    else:
                        topics_qty_year[topic][year] += 1

                if publisher not in articles_by_publisher:
                    articles_by_publisher[publisher] = {}
                    articles_by_publisher[publisher][year] = {}
                    articles_by_publisher[publisher][year][topic] = 1
                elif year in articles_by_publisher[publisher]:
                    if topic not in articles_by_publisher[publisher][year]:
                        articles_by_publisher[publisher][year][topic] = 1
                    else:
                        articles_by_publisher[publisher][year][topic] += 1
                else:
                    articles_by_publisher[publisher][year] = {}
                    articles_by_publisher[publisher][year][topic] = 1

                if publisher not in publisher_qty:
                    publisher_qty[publisher] = {}
                    publisher_qty[publisher][topic] = 1
                elif topic not in publisher_qty[publisher]:
                    publisher_qty[publisher][topic] = 1
                else:
                    publisher_qty[publisher][topic] += 1
    
    return topics_publisher_year, topics_year_publisher, topics_qty, articles_by_publisher, publisher_qty, topics_qty_year, publisher_qty_year, year_topics

def analyse_topics(topics_publisher_year, biggest_topics, topics_year_publisher):
    for topic in biggest_topics:
        topic_name = topics_publisher_year[topic[0]]
        print(topic_name)
        topic_name_df = pd.DataFrame.from_dict(topic_name)
        topic_name_df = topic_name_df.replace(np.nan, 0)
        print(topic_name_df)
        # Get data from last year

        topic_name_last_year = topic_name_df.sort_index().tail(1)
        last_year = topic_name_df.sort_index().tail(1).index.values[0]
        topic_name_last_year = topic_name_last_year.values[0]

        # topic_name_2022 = topic_name_df.sort_index().tail(1).values[0]
        print(topic_name_last_year)
        topic_name_df = topic_name_df.agg(['sum']) # Get total quantity for each conference in this topic
        print(topic_name_df)

        x_data = list(topic_name_df.columns)
        y_data = list(topic_name_df.values[0])
        print(x_data)
        print('------')
        print(y_data)

        comparison_dataframe = pd.DataFrame(data={'Publisher': x_data, 'Total': y_data, f'{last_year} Qty': topic_name_last_year})
        print(comparison_dataframe)

        data = ColumnDataSource(data=dict(x_data=x_data, y_data=y_data))  

        p = figure(x_range=x_data,
                plot_width=1200,
                plot_height=520, 
                #toolbar_location=None, 
                title=f'Artigos publicados no topico {topic[0]}'
                )  # cria figura
        p.vbar(x='x_data', 
            top='y_data', 
            width=0.9, 
            source=data, 
            legend_field="x_data",
            line_color='white',
            fill_color=factor_cmap('x_data', palette=Category10[10], factors=x_data)
            )
        p.xaxis.major_label_orientation = np.math.pi/4   # legend orientation by angle pi/x
        p.legend.location = "top_left"
        show(p)

        dentities[f'et-gridplot_published_articles_{topic[0]}'] = d1.entity(f'ufrj:gridplot_published_articles_{topic[0]}', {'prov:label': f'Plot dos artigos publicados no tópico {topic[0]}', 'prov:type': 'foaf:Document'})
        d1.wasGeneratedBy(dentities[f'et-gridplot_published_articles_{topic[0]}'], dactivities["at-analyse_topics"])

        #
        fig = px.bar(comparison_dataframe,
                    x=comparison_dataframe['Publisher'], 
                    y=comparison_dataframe[f'{last_year} Qty'],
                    hover_data=['Total'], 
                    color='Total',
                    labels={'Publisher':'Publishers'}, #height=400
                    )
        fig.update_layout(title_text=f'Artigos publicados no tópico {topic[0]} em {last_year}')
        fig.update_xaxes(tickangle=-45) 
        fig.show()

        dentities[f'et-gridplot_published_articles_{topic[0]}_{last_year}'] = d1.entity(f'ufrj:gridplot_published_articles_{topic[0]}_{last_year}', {'prov:label': f'Plot dos artigos publicados no tópico {topic[0]} em {last_year}', 'prov:type': 'foaf:Document'})
        d1.wasGeneratedBy(dentities[f'et-gridplot_published_articles_{topic[0]}_{last_year}'], dactivities["at-analyse_topics"])

        topics_data = []
        for year in topics_year_publisher[topic[0]].keys():
            cur_data = []
            for publisher in topics_publisher_year[topic[0]].keys():
                if publisher in topics_year_publisher[topic[0]][year]:
                    cur_data = [year,publisher,topics_year_publisher[topic[0]][year][publisher]]
                    topics_data.append(cur_data)

        print(topics_data)

        topics_data_df = pd.DataFrame(topics_data, columns=['Year', 'Publisher', 'Qty'])

        sns.set_theme()
        sns.relplot(data=topics_data_df, x="Year", y="Qty",
                    hue="Publisher", style="Publisher", size="Qty")
        plt.show()

        dentities[f'et-histogram_published_articles_{topic[0]}_publisher_year'] = d1.entity(f'ufrj:histogram_published_articles_{topic[0]}_publisher_year', {'prov:label': f'Histograma dos artigos publicados no tópico {topic[0]} por conferência e ano', 'prov:type': 'foaf:Document'})
        d1.wasGeneratedBy(dentities[f'et-histogram_published_articles_{topic[0]}_publisher_year'], dactivities["at-analyse_topics"])

def analyse_biggest_topics(biggest_topics, topics_qty_year):
    colors = ['orange','green','blue','red','purple','black']
    p1 = figure(title="Quantidade de artigos por tópico por ano")
    p2 = figure(title="Quantidade total de artigos por tópico")

    for i,topic in enumerate(biggest_topics):
        total_years = []
        topic_qty = dict(sorted(topics_qty_year[topic[0]].items()))

        for year in topic_qty.keys():
            if len(total_years) == 0:
                total_years.append(topic_qty[year])
            else:
                total_years.append(topic_qty[year] + total_years[len(total_years) - 1])

        p1.line(list(topic_qty.keys()), list(topic_qty.values()), legend_label=f"{topic[0]}", line_color=colors[i], line_dash=(4, 4),line_width=2)
        p2.line(list(topic_qty.keys()), list(total_years), legend_label=f"{topic[0]}", line_color=colors[i], line_dash=(4, 4),line_width=2)
    p2.legend.location = "bottom_right"
    show(gridplot([p1, p2], ncols=2, width=600, height=400))

    dentities[f'et-gridplot_topics'] = d1.entity(f'ufrj:gridplot_topics', {'prov:label': f'Plot com artigos publicados por tópico', 'prov:type': 'foaf:Document'}) 

def analyse_most_published_events(publisher_qty_year, publisher_qty_df_total):
    x_data = list(publisher_qty_df_total.columns)
    y_data = list(publisher_qty_df_total.values[0])
    print(y_data)

    data = ColumnDataSource(data=dict(x_data=x_data, y_data=y_data))

    p = figure(x_range=x_data,
            plot_width=1200,
            plot_height=520, 
            title=f"Artigos publicados nos eventos"
            )  # cria figura
    p.vbar(x='x_data', 
        top='y_data', 
        width=0.9, 
        source=data, 
        legend_field="x_data",
        line_color='white',
        fill_color=factor_cmap('x_data', palette=Category10[10], factors=x_data)
        )
    p.xaxis.major_label_orientation = np.math.pi/4   # legend orientation by angle pi/x
    p.legend.location = "top_left"
    show(p)

    dentities['et-gridplot_published_articles_events'] = d1.entity('ufrj:gridplot_published_articles_events', {'prov:label': 'Plot dos artigos publicados nas conferências', 'prov:type': 'foaf:Document'})

    colors = ['orange','green','blue','red','purple','black','pink','yellow','brown','LightSeaGreen']
    p1 = figure(title="Quantidade de artigos por evento por ano")
    p2 = figure(title="Quantidade total de artigos por evento")
    for i,pub in enumerate(x_data):
        total_years = []
        pub_qty = dict(sorted(publisher_qty_year[pub].items()))
        for year in pub_qty.keys():
            if len(total_years) == 0:
                total_years.append(pub_qty[year])
            else:
                total_years.append(pub_qty[year] + total_years[len(total_years) - 1])
        p1.line(list(pub_qty.keys()), list(pub_qty.values()), legend_label=f"{pub}", line_color=colors[i], line_dash=(4, 4),line_width=2)
        p2.line(list(pub_qty.keys()), list(total_years), legend_label=f"{pub}", line_color=colors[i], line_dash=(4, 4),line_width=2)
    show(gridplot([p1, p2], ncols=2, width=1000, height=400))

    dentities['et-histogram_events'] = d1.entity('ufrj:histogram_events', {'prov:label': 'Histograma dos artigos publicados nas conferências por ano e total', 'prov:type': 'foaf:Document'})

def analyse_publishers(publisher_qty_df,topics_qty,articles_by_publisher):
    for event in publisher_qty_df.columns:
        # Sort events from quantity of published articles (most to least)
        sorted_df = publisher_qty_df.sort_values(by=event, ascending=False)
        sorted_df = sorted_df[event].head(6)
        # print(sorted_df)

        total_topic = []
        big_qty_topic = sorted_df.index.to_list()
        # print(topics_qty)
        for topic in big_qty_topic:
            total_topic.append(topics_qty[topic])

        d = {'Topic': big_qty_topic, 'Total': total_topic, f'{event} Qty': list(sorted_df.values)}
        print(d)
        comparison_dataframe = pd.DataFrame(data=d)
        print(comparison_dataframe)

        fig = px.bar(comparison_dataframe,
                    x=comparison_dataframe['Topic'], 
                    y=comparison_dataframe[f'{event} Qty'],
                    hover_data=['Total'], 
                    color='Total',
                    labels={'Topic':'Topic'}, #height=400
                    )
        fig.update_layout(title_text=f'Quantidade de publicacoes dos topicos no {event}')
        fig.update_xaxes(tickangle=-45) 
        fig.show()

        dentities[f'et-plot_topics_event_{event}'] = d1.entity(f'ufrj:plot_topics_event_{event}', {'prov:label': f'Plot das publicações dos tópicos no evento {event}', 'prov:type': 'foaf:Document'})
        d1.wasGeneratedBy(dentities[f'et-plot_topics_event_{event}'], dactivities["at-analyse_publishers"])

        publishing_data = []
        # topic ano qty
        for year in articles_by_publisher[event].keys():
            cur_data = []
            for topic in big_qty_topic:
                if topic in articles_by_publisher[event][year]:
                    cur_data = [year,topic,articles_by_publisher[event][year][topic]]
                    publishing_data.append(cur_data)

        print(publishing_data)

        publishing_data_df = pd.DataFrame(publishing_data, columns=['Year', 'Topic', 'Qty'])

        sns.set_theme()
        sns.relplot(data=publishing_data_df, x="Year", y="Qty",
                    hue="Topic", style="Topic", size="Qty")
        plt.show()

        dentities[f'et-histogram_topics_event_{event}'] = d1.entity(f'ufrj:histogram_topics_event_{event}', {'prov:label': f'Histograma das publicações dos tópicos no evento {event} ao longo dos anos', 'prov:type': 'foaf:Document'})
        d1.wasGeneratedBy(dentities[f'et-histogram_topics_event_{event}'], dactivities["at-analyse_publishers"])

# Find quantity of clusters
def calculate_wcss(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
        #
    return wcss

# Perform clustering on each topic
def find_clusters(biggest_topics, publisher_qty, publisher_qty_df_total):

    # For each topic find the cluster
    for i,topic in enumerate(biggest_topics):
        topic = topic[0]
        print(topic)
        topic_events_qty = []

        start_time = datetime.datetime.now()
   
        # Find the quantity of articles published with this topic in this event
        for event in publisher_qty.keys():
            if topic in publisher_qty[event]:
                topic_events_qty.append(publisher_qty[event][topic])
            else:
                topic_events_qty.append(0)

        end_time = datetime.datetime.now()

        dentities[f'et-topic_events_qty_{topic}'] = d1.entity(f'ufrj:topic_events_qty_{topic}', {'prov:label': f'Lista da quantidade de artigos publicados no tópico {topic} em cada conferência', 'prov:type': 'foaf:Document'})
        dactivities[f'at-get_qty_topic_event_{topic}'] = d1.activity(f'ufrj:get_qty_topic_event_{topic}', start_time, end_time)
        d1.wasAssociatedWith(dactivities[f'at-get_qty_topic_event_{topic}'], dagnts["ag-aa-ipynb"])
        d1.used(dactivities[f'at-get_qty_topic_event_{topic}'], dentities['et-publisher_qty'])
        d1.wasGeneratedBy(dentities[f'et-topic_events_qty_{topic}'], dactivities[f'at-get_qty_topic_event_{topic}']) 

        # Create Dataframe with quantity of articles from topic in event and total quantity of articles in event
        event_topic = pd.DataFrame(data={'Event Topic': topic_events_qty, 'Event': publisher_qty_df_total.values[0]}, index=publisher_qty_df_total.columns)

        dentities[f'et-event_topic_{topic}'] = d1.entity(f'ufrj:event_topic_{topic}', {'prov:label': f'Dataframe com a quantidade de artigos do tópico {topic} no total e em cada conferência', 'prov:type': 'foaf:Document'})
        d1.wasDerivedFrom(dentities[f'et-topic_events_qty_{topic}'], dentities[f'et-event_topic_{topic}'])
        d1.wasDerivedFrom(dentities['et-publisher_qty_df_total'], dentities[f'et-event_topic_{topic}'])


        start_time = datetime.datetime.now()
        # Find quantity of clusters
        wcss = calculate_wcss(event_topic)

        end_time = datetime.datetime.now()

        dentities[f'et-wcss_{topic}'] = d1.entity(f'ufrj:wcss_{topic}', {'prov:label': f'Quantidade de clusters do tópico {topic}', 'prov:type': 'foaf:Document'})
        dactivities[f'at-calculate_wcss_{topic}'] = d1.activity(f'ufrj:calculate_wcss_{topic}', start_time, end_time)
        d1.wasAssociatedWith(dactivities[f'at-calculate_wcss_{topic}'], dagnts["ag-aa-ipynb"])
        d1.used(dactivities[f'at-calculate_wcss_{topic}'], dentities[f'et-event_topic_{topic}'])
        d1.wasGeneratedBy(dentities[f'et-wcss_{topic}'], dactivities[f'at-calculate_wcss_{topic}']) 

        fig = plt.figure(figsize=(6,4))
        plt.plot(range(1, 11), wcss, 'r', lw=2.0)
        plt.title('Método de Elbow')
        plt.xlabel('Número de clusters')
        plt.ylabel('WCSS')
        plt.grid()
        plt.show()

        # Initialize the clusters
        #sns.set()
        start_time = datetime.datetime.now()
        kmeans = KMeans(n_clusters = 3, init ='k-means++', max_iter=300, n_init = 10, random_state=0)
        #

        # Get values from Dataframe to plot clusters
        event_topic = event_topic.values
        
        y_kmeans = kmeans.fit_predict(event_topic)        # Adjust the clusters
        # figure
        cluster = plt.figure(figsize=(15,10))
        #
        plt.scatter(event_topic[y_kmeans == 0, 0], event_topic[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(event_topic[y_kmeans == 1, 0], event_topic[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(event_topic[y_kmeans == 2, 0], event_topic[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        #
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c = 'black', label = 'Centroides')
        plt.title('Clusters de consumidores')
        plt.xlabel(f'Quantidade do tópico {topic}')
        plt.ylabel('Quantidade total')
        plt.legend()
        plt.show()

        end_time = datetime.datetime.now()

        dentities[f'et-cluster_image_{topic}'] = d1.entity(f'ufrj:cluster_image_{topic}', {'prov:label': f'Plot de clusters para {topic}', 'prov:type': 'foaf:Document'})
        dactivities[f'at-plot_image_{topic}'] = d1.activity(f'ufrj:plot_image_{topic}', start_time, end_time)
        d1.wasAssociatedWith(dactivities[f'at-plot_image_{topic}'], dagnts["ag-aa-ipynb"])
        d1.used(dactivities[f'at-plot_image_{topic}'], dentities[f'et-event_topic_{topic}'])
        d1.wasGeneratedBy(dentities[f'et-cluster_image_{topic}'], dactivities[f'at-plot_image_{topic}'])  

        cluster.savefig(f'Imagens/cluster-{topic}.png')

def find_biggest6_per_year(year_topics):
    biggest_topics_year = {}

    for year in year_topics.keys():
        sorted_year_topics = dict(sorted(year_topics[year].items(), key=lambda item: item[1], reverse=True))
        biggest_topics_year[year] = list(sorted_year_topics.items())[0:6]

    print(biggest_topics_year)
    return biggest_topics_year

def main(article_data):

    start_time = datetime.datetime.now()
    topics_publisher_year, topics_year_publisher, topics_qty, articles_by_publisher, publisher_qty, topics_qty_year, publisher_qty_year, year_topics = create_dicts(article_data)
    end_time = datetime.datetime.now()

    dentities['et-topics_publisher_year'] = d1.entity('ufrj:topics_publisher_year', {'prov:label': 'Quantidade de artigos publicados por cada conferência em cada ano por tópico', 'prov:type': 'foaf:Document'})
    dentities['et-topics_year_publisher'] = d1.entity('ufrj:topics_year_publisher', {'prov:label': 'Quantidade de artigos publicados por cada ano em cada conferência por tópico', 'prov:type': 'foaf:Document'})
    dentities['et-topics_qty'] = d1.entity('ufrj:topics_qty', {'prov:label': 'Quantidade de artigos publicados de cada tópico', 'prov:type': 'foaf:Document'})
    dentities['et-articles_by_publisher'] = d1.entity('ufrj:articles_by_publisher', {'prov:label': 'Quantidade de artigos publicados por cada ano em cada tópico por conferência', 'prov:type': 'foaf:Document'})
    dentities['et-publisher_qty'] = d1.entity('ufrj:publisher_qty', {'prov:label': 'Quantidade de vezes que cada conferência publicou cada tópico', 'prov:type': 'foaf:Document'})
    dentities['et-topics_qty_year'] = d1.entity('ufrj:topics_qty_year', {'prov:label': 'Quantidade de vezes que cada tópico foi publicado em cada ano', 'prov:type': 'foaf:Document'})
    dentities['et-publisher_qty_year'] = d1.entity('ufrj:publisher_qty_year', {'prov:label': 'Quantidade de publicações de cada conferência por ano', 'prov:type': 'foaf:Document'})
    dentities['et-year_topics'] = d1.entity('ufrj:year_topics', {'prov:label': 'Quantidade de vezes que um tópico apareceu por ano', 'prov:type': 'foaf:Document'})

    dactivities["at-create_dicts"] = d1.activity("ufrj:create_dicts", start_time, end_time)
    d1.wasAssociatedWith(dactivities['at-create_dicts'], dagnts["ag-aa-ipynb"])
    d1.used(dactivities["at-create_dicts"], dentities['et-dataset'])
    d1.wasGeneratedBy(dentities['et-topics_publisher_year'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-topics_year_publisher'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-topics_qty'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-articles_by_publisher'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-publisher_qty'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-topics_qty_year'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-publisher_qty_year'], dactivities["at-create_dicts"])
    d1.wasGeneratedBy(dentities['et-year_topics'], dactivities["at-create_dicts"])
    
    # Sort topics by total quantity published
    topics_qty = dict(sorted(topics_qty.items(), key=lambda item: item[1], reverse=True))

    dentities['et-sorted_topics_qty'] = d1.entity('ufrj:sorted_topics_qty', {'prov:label': 'Quantidade ordenada de artigos publicados de cada tópico', 'prov:type': 'foaf:Document'})

    d1.wasDerivedFrom(dentities['et-sorted_topics_qty'], dentities['et-topics_qty'])

    # Get 6 biggest topics
    biggest_topics = list(topics_qty.items())[0:6]

    dentities['et-biggest_topics'] = d1.entity('ufrj:biggest_topics', {'prov:label': 'Lista com os 6 tópicos que mais apareceram nos artigos', 'prov:type': 'foaf:Document'})

    d1.wasDerivedFrom(dentities['et-biggest_topics'], dentities['et-sorted_topics_qty'])
    # print('Biggest Topics: ')
    # print(biggest_topics)

    start_time = datetime.datetime.now()
    biggest_topics_year = find_biggest6_per_year(year_topics)
    end_time = datetime.datetime.now()

    dactivities["at-find_biggest6_per_year"] = d1.activity("ufrj:find_biggest6_per_year", start_time, end_time)
    d1.wasAssociatedWith(dactivities['at-find_biggest6_per_year'], dagnts["ag-aa-ipynb"])
    dentities['et-biggest_topics_year'] = d1.entity('ufrj:biggest_topics_year', {'prov:label': 'Dicionário com as publicações dos 6 tópicos mais publicados por ano', 'prov:type': 'foaf:Document'})
    d1.used(dactivities["at-find_biggest6_per_year"], dentities['et-biggest_topics'])
    d1.wasGeneratedBy(dentities['et-biggest_topics_year'], dactivities["at-find_biggest6_per_year"])

    # Get events with biggest quantity of published articles
    publisher_qty_year_df = pd.DataFrame.from_dict(publisher_qty_year).replace(np.nan, 0)
    dentities['et-publisher_qty_year_df'] = d1.entity('ufrj:publisher_qty_year_df', {'prov:label': 'Dataframe com quantidade de publicações de cada conferência por ano', 'prov:type': 'foaf:Document'})
    d1.wasDerivedFrom(dentities['et-publisher_qty_year'], dentities['et-publisher_qty_year_df'])

    publisher_qty_df = pd.DataFrame.from_dict(publisher_qty).replace(np.nan, 0)
    dentities['et-publisher_qty_df'] = d1.entity('ufrj:publisher_qty_year_df', {'prov:label': 'Dataframe com quantidade de vezes que cada conferência publicou cada tópico', 'prov:type': 'foaf:Document'})
    d1.wasDerivedFrom(dentities['et-publisher_qty'], dentities['et-publisher_qty_df'])

    publisher_qty_df_total = publisher_qty_year_df.agg(['sum'])
    dentities['et-publisher_qty_df_total'] = d1.entity('ufrj:publisher_qty_df_total', {'prov:label': 'Soma da quantidade de publicações de cada conferência ', 'prov:type': 'foaf:Document'})
    d1.wasDerivedFrom(dentities['et-publisher_qty_year_df'], dentities['et-publisher_qty_df_total'])
    # print(publisher_qty_df_total)

    # Perform clustering on each topic
    find_clusters(biggest_topics, publisher_qty, publisher_qty_df_total)

    dactivities["at-analyse_topics"] = d1.activity("ufrj:analyse_topics")
    d1.wasAssociatedWith(dactivities['at-analyse_topics'], dagnts["ag-aa-ipynb"])
    analyse_topics(topics_publisher_year,biggest_topics,topics_year_publisher)

    d1.used(dactivities["at-analyse_topics"], dentities['et-topics_publisher_year'])
    d1.used(dactivities["at-analyse_topics"], dentities['et-biggest_topics'])
    d1.used(dactivities["at-analyse_topics"], dentities['et-topics_year_publisher'])

    start_time = datetime.datetime.now()
    analyse_biggest_topics(biggest_topics,topics_qty_year)
    end_time = datetime.datetime.now()

    dactivities["at-analyse_biggest_topics"] = d1.activity("ufrj:analyse_biggest_topics", start_time, end_time)
    d1.wasAssociatedWith(dactivities['at-analyse_biggest_topics'], dagnts["ag-aa-ipynb"])
    d1.used(dactivities["at-analyse_biggest_topics"], dentities['et-biggest_topics'])
    d1.used(dactivities["at-analyse_biggest_topics"], dentities['et-topics_qty_year'])
    d1.wasGeneratedBy(dentities['et-gridplot_topics'], dactivities["at-analyse_biggest_topics"])

    start_time = datetime.datetime.now()
    analyse_most_published_events(publisher_qty_year, publisher_qty_df_total)
    end_time = datetime.datetime.now()

    dactivities["at-analyse_most_published_events"] = d1.activity("ufrj:analyse_most_published_events", start_time, end_time)
    d1.wasAssociatedWith(dactivities['at-analyse_most_published_events'], dagnts["ag-aa-ipynb"])
    d1.used(dactivities["at-analyse_most_published_events"], dentities['et-publisher_qty_year'])
    d1.used(dactivities["at-analyse_most_published_events"], dentities['et-publisher_qty_df_total'])
    d1.wasGeneratedBy(dentities['et-gridplot_published_articles_events'], dactivities["at-analyse_most_published_events"])
    d1.wasGeneratedBy(dentities['et-histogram_events'], dactivities["at-analyse_most_published_events"])

    dactivities["at-analyse_publishers"] = d1.activity("ufrj:analyse_publishers")
    d1.wasAssociatedWith(dactivities['at-analyse_publishers'], dagnts["ag-aa-ipynb"])
    analyse_publishers(publisher_qty_df,topics_qty,articles_by_publisher)

    d1.used(dactivities["at-analyse_publishers"], dentities['et-publisher_qty_df'])
    d1.used(dactivities["at-analyse_publishers"], dentities['et-sorted_topics_qty'])
    d1.used(dactivities["at-analyse_publishers"], dentities['et-articles_by_publisher'])

def create_agents(d1,agents):
    agents["ag-ufrj"] = d1.agent("ufrj:UFRJ", {"prov:type":"prov:Organization", "foaf:name":"Universidade Federal do Rio de Janeiro"})
    agents["ag-ppgi"] = d1.agent("ufrj:PPGI", {"prov:type":"prov:Organization", "foaf:name":"Programa de Pós Graduação em Informática"})
    agents["ag-ppgi"].actedOnBehalfOf(agents["ag-ufrj"])
    agents["ag-mai712"] = d1.agent("ufrj:MAI712", {"prov:type":"prov:Organization", "foaf:name":"Disciplina de Fundamentos de Ciências de Dados"})
    agents["ag-mai712"].actedOnBehalfOf(agents["ag-ppgi"])
    agents["ag-grupo6"] = d1.agent("ufrj:GRUPO6", {"prov:type":"prov:Organization", "foaf:name":"Grupo 06 para o trabalho final"})
    agents["ag-grupo6"].actedOnBehalfOf(agents["ag-mai712"])
    agents["ag-aluno-ingrid"] = d1.agent("ufrj:Ingrid", {"prov:type":"foaf:Person", "foaf:name":"Ingrid Quintanilha Pacheco", "foaf:mbox":"ingrid.pacheco@dcc.ufrj.br"})
    agents["ag-aluno-ingrid"].actedOnBehalfOf(agents["ag-grupo6"])
    agents["ag-aluno-eduardo"] = d1.agent("ufrj:Eduardo", {"prov:type":"foaf:Person", "foaf:name":"Eduardo Prata", "foaf:mbox":"edu.prata@gmail.com"})
    agents["ag-aluno-eduardo"].actedOnBehalfOf(agents["ag-grupo6"])
    agents["ag-aa-ipynb"] = d1.agent("ufrj:articleAnalysis.ipynb", {"prov:type":"prov:SoftwareAgent", "foaf:name":"articleAnalysis.ipynb", "prov:label":"Notebook Python usado para análise dos artigos"})
    agents["ag-aa-ipynb"].actedOnBehalfOf(agents["ag-grupo6"])

if __name__ == "__main__":

    d1.add_namespace('cornel', 'https://www.cs.cornell.edu/andru/')
    d1.add_namespace('dblp', 'https://dblp.org/')
    d1.add_namespace('ufrj', 'https://www.ufrj.br')
    d1.add_namespace('foaf', 'http://xmlns.com/foaf/0.1/')

    create_agents(d1,dagnts)

    dentities['et-dataset'] = d1.entity(f'ufrj:articles-2022-09-19.csv', {'prov:label': 'CSV de dataset com os dados dos artigos', 'prov:type': 'foaf:Document'})
    start_time = datetime.datetime.now()

    urllib.request.urlretrieve('https://drive.google.com/uc?id=1kWTbqT4QXZ2cVIP5dIHbgVI6ipRc3ba1&authuser=0&export=download', 'articles-2022-09-19.csv')
    os.listdir()

    article_data = pd.read_csv('articles-2022-09-19.csv',sep = ';')
    article_data.head()
    main(article_data)

    end_time = datetime.datetime.now()
    dactivities["at-analisar-artigos"] = d1.activity("ufrj:analisar-artigos", start_time, end_time)
    d1.used(dactivities["at-analisar-artigos"], dentities['et-dataset'])
    d1.wasAssociatedWith(dactivities["at-analisar-artigos"], dagnts["ag-grupo6"])

    print(d1.get_provn())
    dot = prov_to_dot(d1)
    dot.write_png('./articleAnalysis-prov.png')
    Image('./articleAnalysis-prov.png')