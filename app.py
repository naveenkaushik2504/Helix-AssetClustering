from flask import Flask, render_template, request, redirect, url_for, make_response
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# def create_clusters(df):
#     documents = list(df['Clustering_Equipment Class'])
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(documents)

#     true_k = 2
#     model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#     model.fit(X)

#     print("Top terms per cluster:")
#     order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#     terms = vectorizer.get_feature_names()
#     for i in range(true_k):
#         print("Cluster %d:" % i),
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind])
#     Y = vectorizer.transform(df['Clustering_Equipment Class'])
#     prediction = model.predict(Y)
#     df['cluster_label'] = prediction
#     df.to_csv('test.csv', index=False)
#     return df

def create_clusters(df):
    df_cost = pd.DataFrame(df.groupby(['UTILITA ORDER', 'Clustering_Equipment Class'])\
        .sum()['Sum of Value']).reset_index()
    df_gr_cost = pd.DataFrame(df_cost.groupby(['Clustering_Equipment Class']).mean()['Sum of Value']).reset_index()

    kmeans = KMeans(n_clusters=4, random_state=0)
    labels = kmeans.fit_predict(np.reshape(np.array(df_gr_cost['Sum of Value']), (-1,1)))
    df_gr_cost['cluster_label'] = labels
    df_gr_cost.to_csv('clusters.csv', index=False)
    return df_gr_cost


def create_wc(df):

    for filename in os.listdir('static/'):
        os.remove('static/' + filename)
    
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in df['cluster_label'].unique():
        data = Counter(list(df[df['cluster_label']==i]['Clustering_Equipment Class']))
        wordcloud.generate_from_frequencies(data)
        new_graph_name = "cluster_" + str(i)
        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig('static/' + new_graph_name)

def read_data(file_name):
    df = pd.read_csv(file_name)
    print(df.shape)
    print(df.head())
    df = create_clusters(df)
    create_wc(df)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/results")
def results():
    graphs = os.listdir('static')
    print(graphs)
    return render_template('results.html', hists = graphs)

@app.route('/results', methods=['POST'])
def download_data():
    df = pd.read_csv('clusters.csv')
    print(df.shape)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=clusters.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    read_data(uploaded_file.filename)
    return redirect(url_for('results'))