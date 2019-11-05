import numpy as np
import pandas as pd
import os
from bs4 import BeautifulSoup

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

def vectorize():
    """

    """
    # Here, we set our tokenizer for vectorization. This is an imperfect process, but the RegexpTokenizer seems to work the best at stripping out unnecessary chars, despite the fact that it splits nouns with hyphens and apostrophes:
    tokenizer = RegexpTokenizer(r"\w+")

    # Setting our stopwords for vectorization:
    additional_stopwords = ["episode", "production", "star", "trek", "aren'", "couldn'", "didn'",
                            "doesn'", "don'", "hadn'", "hasn'", "haven'", "isn'", "it'", "mightn'", "mustn'", "needn'", "shan'", "she'", "should'", "shouldn'", "that'", "wasn'", "weren'", "won'", "wouldn'", "you'"]

    stopwords_ = set(stopwords.words("english") + additional_stopwords)

    # Setting up our vectorization method, which takes in the above variables:
    vectorizer = TfidfVectorizer(strip_accents="ascii",
                                 lowercase=True,
                                 tokenizer=test_tokenizer.tokenize,
                                 stop_words=stopwords_
                                 )

    # Here, we set our X to be the text we're looking to cluster; in this instance, the ultra_blob column of our DF. We want ALL of it:
    X = test_vectorizer.fit_transform(combined_master_df["ultra_blob"])
    print(f"Shape of X: {not_X.shape}")
    print(f"X as dense matrix: {not_X.todense()}")
    return X

def get_silhouette_score(max_clusters=24):
    """
    Determines the optimal number of clusters (default 24) and returns silhouette score graph. Some human judgment is going to be necessary to judge whether the highest scoring peaks are indeed the optimal number of max clusters, so some domain-specific knowledge may be necessary.
    """
    maxk = max_clusters
    silhouette = np.zeros(maxk)

    for k in range(1,maxk):
        km = KMeans(n_clusters=k,
                    n_jobs=-1)
        y = km.fit_predict(X)
        
        if k > 1:
            silhouette[k] = silhouette_score(X,y)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(range(2,maxk), silhouette[2:maxk], 'o-', linewidth=2.5, color="aquamarine")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("silhouette score")
    plt.show()
    plt.savefig("data/silhouette_score_graph.png")
