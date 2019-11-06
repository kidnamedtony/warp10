import numpy as np
import pandas as pd
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def create_df_for_vectorization():
    """
    Function to create dataframe for vectorization. Returns a dataframe for that task.
    """
    # Pull in the two DataFrames we want to merge to create the "ultra_blob" column, which has all the text data we want to cluster:
    combined_master_df = pd.read_pickle("data/combined_info_rev_df.pkl.bz2")
    combined_summary_df = pd.read_pickle("data/combined_summary_df.pkl.bz2")

    # We'll be working from the combined master DF, so here we build our ultra_blob column with all of our text data (user reviews, synopses, and fan wikia page contents) smashed together:
    combined_master_df = combined_master_df.merge(combined_summary_df[["identifier", "summary"]], on="identifier")
    combined_master_df["ultra_blob"] = combined_master_df["super_blob"] + " " + combined_master_df["summary"]
    return combined_master_df

def vectorize(max_features=None):
    """
    Function to vectorize the word blobs in our DF. Returns X and features arrays.
    """
    # Here, we set our tokenizer for vectorization. This is an imperfect process, but the RegexpTokenizer seems to work the best at stripping out unnecessary chars, despite the fact that it splits nouns with hyphens and apostrophes:
    tokenizer = RegexpTokenizer(r"\w+")

    # Setting our stopwords for vectorization:
    additional_stopwords = ["episode", "production", "star", "trek", "aren'", "couldn'", "didn'",
                            "doesn'", "don'", "hadn'", "hasn'", "haven'", "isn'", "it'", "mightn'", "mustn'", "needn'", "shan'", "she'", "should'", "shouldn'", "that'", "wasn'", "weren'", "won'", "wouldn'", "you'", "also", "like"]

    stopwords_ = set(stopwords.words("english") + additional_stopwords)

    # Setting up our vectorization method, which takes in the above variables:
    vectorizer = TfidfVectorizer(strip_accents="ascii",
                                 lowercase=True,
                                 tokenizer=test_tokenizer.tokenize,
                                 stop_words=stopwords_,
                                 max_features=None
                                 )

    # Here, we set our X to be the text we're looking to cluster; in this instance, the ultra_blob column of our DF. We want ALL of it:
    X = vectorizer.fit_transform(combined_master_df["ultra_blob"])

    features = vectorizer.get_feature_names()
    print(f"Shape of X: {not_X.shape}")
    print(f"X as dense matrix: {not_X.todense()}")
    return X, features

def get_silhouette_score(max_clusters=24):
    """
    Determines the optimal number of clusters (default 24) and returns silhouette score graph. Some human judgment is going to be necessary to judge whether the highest scoring peaks are indeed the optimal number of max clusters, so some domain-specific knowledge may be necessary.
    """
    maxk = max_clusters
    silhouette = np.zeros(maxk)

    for k in range(1,maxk):
        km = KMeans(n_clusters=k,
                    n_jobs=-1,
                    verbose=1)
        y = km.fit_predict(X)

        if k > 1:
            silhouette[k] = silhouette_score(X,y)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(range(2,maxk), silhouette[2:maxk], 'o-', linewidth=2.5, color="aquamarine")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("silhouette score")
    plt.show()
    plt.savefig("data/silhouette_score_graph.png")

def clusterize(opt_clusters=13, rand_state=0, imdb_rating_threshhold=7.0, X, features_arr):
    """
    Clustering algorithm to...cluster all your vectorized terms. Optimal clusters and random state have default values set based on optimal clustering of Star Trek text data.

    Prints semi-random recommendation of episodes from each cluster, so long as they're rated above a user-defined threshold. Default is set to 7.0, however.

    Input:
    ---
    *opt_clusters: int; optimal number of clusters you want the data to be split into
    *rand_state:int; (optional) random seed. Setting this should split the data consistently whenever this k-means is run again.
    *imdb_rating_threshhold: float; default set to 7.0, assuming that that's a decent enough rating for recommendations to be made.
    *X: array of data to be clustered.
    *features_arr: array of features from vectorizer.

    Output:
    ---
    Prints top 15 words per cluster, along with a random episode at or above the rating threshhold set as a parameter, above, for each cluster.
    """
    kmeans_clusterer = KMeans(n_clusters=opt_clusters,
                              n_jobs=-1,
                              verbose=1,
                              random_state=rand_state
                              )
    kmeans_clusterer.fit(X)

    # After the data is fit to the vectorizer, we can determine our cluster centers and their top 15 terms:
    centroid_array = kmeans_clusterer.cluster_centers_
    top_centroids = centroid_array.argsort()[:, -1:-16:-1]
    print("Top 15 words for each cluster:\n")
    for num, centroid in enumerate(top_centroids):
        print(f"Centroid {num} :",", ".join(features[i] for i in centroid),"\n")

    print(f"A random GOOD ({imdb_rating_threshhold}+ rating) episode in each cluster:\n")
    assigned_cluster = kmeans_clusterer.transform(X).argmin(axis=1)
    for i in range(kmeans_clusterer.n_clusters):
        cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
        ep_title_proposal = np.random.choice(cluster, 1, replace=False)
        if combined_master_df.loc[ep_title_proposal, "IMDB_user_rating"].values >= imdb_rating_threshhold:
            sample_ep_titles = ep_title_proposal
        else:
            continue
        print("Cluster %d:" % i)
        for ep_title in sample_ep_titles:
            print(f'{combined_master_df.loc[ep_title]["series"]} S:{combined_master_df.loc[ep_title]["season"]}E:{combined_master_df.loc[ep_title]["episode"]} {combined_master_df.loc[ep_title]["ep_title"]},  {combined_master_df.loc[ep_title]["IMDB_user_rating"]}')


    # # Print a random selection of episodes from each cluster:
    # print("Random sample of episodes in each cluster:\n")
    # assigned_cluster = kmeans_clusterer.transform(X).argmin(axis=1)
    # for i in range(kmeans_clusterer.n_clusters):
    #     cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
    #     sample_ep_titles = np.random.choice(cluster, 3, replace=False)
    #     print("Cluster %d:" % i)
    #     for ep_title in sample_ep_titles:
    #         print(f'{combined_master_df.loc[ep_title]["series"]}S:{combined_master_df.loc[ep_title]["season"]}E:{combined_master_df.loc[ep_title]["season"]} {combined_master_df.loc[ep_title]["ep_title"]},  {combined_master_df.loc[ep_title]["IMDB_user_rating"]}')
