from cmath import nan
from numpy import NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bokeh.palettes import magma, Category10  
from bokeh.plotting import figure, show            # paleta de cores
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
import plotly.express as px
import seaborn as sns

# Top 6 tópicos mais publicados OK
# Para cada conf, qual o tópico que mais aparece OK
# Qtdade de artigos publicados para cada conf OK
# Top 5 dos 6 maiores tópicos - geral e por ano OK
# Análise histórica de tópicos para cada conf OK
# Análise histórica das confs para cada tópico OK

# Próximos:
    # Clusterização por tema - total vs qtdade na conf
    # Previsão p/ o proximo ano?
    # Fazer analise de pais dos eventos? Temos essa info?

def create_dicts(article_data):
    topics_publisher = {}
    topics_year = {}
    topics_qty = {}
    articles_by_publisher = {}
    publisher_qty = {}

    for i,article in enumerate(article_data.values):
        print('qty: ' + str(i))
        print('artigo: ' + article[0])
        print('topics: ')
        topics = article[5]
        print(topics)
        publisher = article[4]
        year = article[7]
        if not pd.isna(publisher) and not pd.isna(topics):
            for topic in topics.split(','):
                topic = topic.upper()
                if topic not in topics_publisher:
                    topics_publisher[topic] = {}
                    topics_publisher[topic][publisher] = {}
                    topics_publisher[topic][publisher][year] = 1
                elif publisher in topics_publisher[topic]:
                    if year not in topics_publisher[topic][publisher]:
                        topics_publisher[topic][publisher][year] = 1
                    else:
                        topics_publisher[topic][publisher][year] += 1
                else:
                    topics_publisher[topic][publisher] = {}
                    topics_publisher[topic][publisher][year] = 1
                
                if topic not in topics_year:
                    topics_year[topic] = {}
                    topics_year[topic][year] = {}
                    topics_year[topic][year][publisher] = 1
                elif year in topics_year[topic]:
                    if publisher not in topics_year[topic][year]:
                        topics_year[topic][year][publisher] = 1
                    else:
                        topics_year[topic][year][publisher] += 1
                else:
                    topics_year[topic][year] = {}
                    topics_year[topic][year][publisher] = 1

                if topic not in topics_qty:
                    topics_qty[topic] = 1
                else:
                    topics_qty[topic] += 1

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
    
    return topics_publisher, topics_year, topics_qty, articles_by_publisher, publisher_qty

def analyse_topics(topics_publisher, biggest_topics, topics_year):
    for topic in biggest_topics:
        topic_name = topics_publisher[topic[0]]
        print(topic_name)
        topic_name_df = pd.DataFrame.from_dict(topic_name)
        topic_name_df = topic_name_df.replace(np.nan, 0)
        print(topic_name_df)
        topic_name_2022 = topic_name_df.tail(1).values[0]
        print(topic_name_2022)
        topic_name_df = topic_name_df.agg(['sum'])
        print(topic_name_df)

        x_data = list(topic_name_df.columns)
        y_data = list(topic_name_df.values[0])
        print(x_data)
        print('------')
        print(y_data)

        d = {'Publisher': x_data, 'Total': y_data, '2022 Qty': topic_name_2022}
        comparison_dataframe = pd.DataFrame(data=d)
        print(comparison_dataframe)

        data = ColumnDataSource(data=dict(x_data=x_data, y_data=y_data))  

        p = figure(x_range=x_data,
                plot_width=1200,
                plot_height=520, 
                #toolbar_location=None, 
                title=f"Artigos publicados no topico {topic[0]}"
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
                    y=comparison_dataframe['2022 Qty'],
                    hover_data=['Total'], 
                    color='Total',
                    labels={'Publisher':'Publishers'}, #height=400
                    )
        fig.update_layout(title_text=f'Artigos publicados no tópico {topic[0]} em 2022')
        fig.update_xaxes(tickangle=-45) 
        fig.show()

        topics_data = []
        for year in topics_year[topic[0]].keys():
            cur_data = []
            for publisher in topics_publisher[topic[0]].keys():
                if publisher in topics_year[topic[0]][year]:
                    cur_data = [year,publisher,topics_year[topic[0]][year][publisher]]
                    topics_data.append(cur_data)

        print(topics_data)

        topics_data_df = pd.DataFrame(topics_data, columns=['Year', 'Publisher', 'Qty'])

        sns.set_theme()
        sns.relplot(data=topics_data_df, x="Year", y="Qty",
                    hue="Publisher", style="Publisher", size="Qty")
        plt.show()

def analyse_most_published_events(publisher_qty):
    publisher_qty_df = pd.DataFrame.from_dict(publisher_qty)
    publisher_qty_df = publisher_qty_df.replace(np.nan, 0)
    print(publisher_qty_df)

    # Get events with biggest quantity of published articles
    publisher_qty_df_total = publisher_qty_df.agg(['sum'])
    print(publisher_qty_df_total)

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
    p.legend.location = "top_center"
    show(p)

    return publisher_qty_df

def analyse_publishers(publisher_qty_df,topics_qty,articles_by_publisher):
    for event in publisher_qty_df.columns:
        sorted_df = publisher_qty_df.sort_values(by=event, ascending=False)
        sorted_df = sorted_df[event].head(5)
        print(sorted_df)

        total_topic = []
        big_qty_topic = sorted_df.index.to_list()
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

def main(article_data):

    topics_publisher, topics_year, topics_qty, articles_by_publisher, publisher_qty = create_dicts(article_data)

    # Sort topics by total quantity published
    topics_qty = dict(sorted(topics_qty.items(), key=lambda item: item[1], reverse=True))
    # Get 6 biggest topics
    biggest_topics = list(topics_qty.items())[0:6]
    print('Biggest Topics: ')
    print(biggest_topics)

    analyse_topics(topics_publisher,biggest_topics,topics_year)

    publisher_qty_df = analyse_most_published_events(publisher_qty)

    analyse_publishers(publisher_qty_df,topics_qty,articles_by_publisher)

if __name__ == "__main__":
    article_data = pd.read_csv('Dataset/articles-2022-09-19.csv',sep = ';')
    article_data.head()
    main(article_data)