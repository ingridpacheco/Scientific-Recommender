import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bokeh.palettes import Category10  
from bokeh.plotting import figure, show            # paleta de cores
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans

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
            #fill_color=factor_cmap('x_data', palette=viridis(10), factors=x_data)  # cores
            #fill_color=factor_cmap('x_data', palette=magma(10), factors=x_data)
            #fill_color=factor_cmap('x_data', palette=Spectral6[10], factors=x_data)   # 6 cores
            fill_color=factor_cmap('x_data', palette=Category10[10], factors=x_data)
            )
        p.xaxis.major_label_orientation = np.math.pi/4   # legend orientation by angle pi/x
        p.legend.location = "top_left"
        show(p)
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

def analyse_most_published_events(publisher_qty_year, publisher_qty_df_total):
    x_data = list(publisher_qty_df_total.columns)
    y_data = list(publisher_qty_df_total.values[0])
    print(y_data)

    data = ColumnDataSource(data=dict(x_data=x_data, y_data=y_data))

    p = figure(x_range=x_data,
            plot_width=1200,
            plot_height=520, 
            #toolbar_location=None, 
            title=f"Artigos publicados nos eventos"
            )  # cria figura
    p.vbar(x='x_data', 
        top='y_data', 
        width=0.9, 
        source=data, 
        legend_field="x_data",
        line_color='white',
        #fill_color=factor_cmap('x_data', palette=viridis(10), factors=x_data)  # cores
        #fill_color=factor_cmap('x_data', palette=magma(10), factors=x_data)
        #fill_color=factor_cmap('x_data', palette=Spectral6[10], factors=x_data)   # 6 cores
        fill_color=factor_cmap('x_data', palette=Category10[10], factors=x_data)
        )
    p.xaxis.major_label_orientation = np.math.pi/4   # legend orientation by angle pi/x
    p.legend.location = "top_left"
    show(p)

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

        # Find the quantity of articles published with this topic in this event
        for event in publisher_qty.keys():
            if topic in publisher_qty[event]:
                topic_events_qty.append(publisher_qty[event][topic])
            else:
                topic_events_qty.append(0)

        # Create Dataframe with quantity of articles from topic in event and total quantity of articles in event
        event_topic = pd.DataFrame(data={'Event Topic': topic_events_qty, 'Event': publisher_qty_df_total.values[0]}, index=publisher_qty_df_total.columns)

        # Find quantity of clusters
        wcss = calculate_wcss(event_topic)

        fig = plt.figure(figsize=(6,4))
        plt.plot(range(1, 11), wcss, 'r', lw=2.0)
        plt.title('Método de Elbow')
        plt.xlabel('Número de clusters')
        plt.ylabel('WCSS')
        plt.grid()
        plt.show()

        # Initialize the clusters
        #sns.set()
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
        cluster.savefig(f'Imagens/cluster-{topic}.png')

def find_biggest6_per_year(year_topics):
    biggest_topics_year = {}

    for year in year_topics.keys():
        sorted_year_topics = dict(sorted(year_topics[year].items(), key=lambda item: item[1], reverse=True))
        biggest_topics_year[year] = list(sorted_year_topics.items())[0:6]

    print(biggest_topics_year)
    return biggest_topics_year

def main(article_data):

    topics_publisher_year, topics_year_publisher, topics_qty, articles_by_publisher, publisher_qty, topics_qty_year, publisher_qty_year, year_topics = create_dicts(article_data)

    # Sort topics by total quantity published
    topics_qty = dict(sorted(topics_qty.items(), key=lambda item: item[1], reverse=True))

    # Get 6 biggest topics
    biggest_topics = list(topics_qty.items())[0:6]
    # print('Biggest Topics: ')
    # print(biggest_topics)

    find_biggest6_per_year(year_topics)

    # Get events with biggest quantity of published articles
    publisher_qty_year_df = pd.DataFrame.from_dict(publisher_qty_year).replace(np.nan, 0)
    publisher_qty_df = pd.DataFrame.from_dict(publisher_qty).replace(np.nan, 0)
    publisher_qty_df_total = publisher_qty_year_df.agg(['sum'])
    # print(publisher_qty_df_total)

    # Perform clustering on each topic
    find_clusters(biggest_topics, publisher_qty, publisher_qty_df_total)

    analyse_topics(topics_publisher_year,biggest_topics,topics_year_publisher)
    analyse_biggest_topics(biggest_topics,topics_qty_year)

    analyse_most_published_events(publisher_qty_year, publisher_qty_df_total)

    analyse_publishers(publisher_qty_df,topics_qty,articles_by_publisher)

if __name__ == "__main__":
    article_data = pd.read_csv('Dataset/articles-2022-09-19.csv',sep = ';')
    article_data.head()
    main(article_data)