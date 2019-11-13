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

def clusterize(opt_clusters=13, rand_state=8, imdb_rating_threshhold=7.0, max_feat=None):
    """
    Clustering algorithm to...cluster all your vectorized terms. Optimal clusters and random state have default values set based on optimal clustering of Star Trek text data.

    Prints semi-random episode recommendations from each cluster and returns a DF of those recommendations, so long as the episode recs are rated above a user-defined threshold. Default is set to 7.0, however.

    Input:
    ---
    *opt_clusters: int; optimal number of clusters you want the data to be split into
    *rand_state:int; (optional) random seed. Setting this should split the data consistently whenever this k-means is run again. Default is set to 8, as that (subjectively) has proven to be a decent clustering.
    *imdb_rating_threshhold: float; default set to 7.0, assuming that that's a decent enough rating for recommendations to be made.
    *max_feat: int; determines the maximum number of features (or terms) for vectorization.

    Output:
    ---
    Prints top 15 words per cluster, along with a random episode at or above the rating threshhold set as a parameter, above, for each cluster.
    """
    # Create the master DF, from which we'll vectorize its "ultra_blob" text column:
    combined_master_df = create_df_for_vectorization()

    # Following on from the previous line, here is where we call the vectorize function on the DF:
    X, features_arr = vectorize(combined_master_df, max_features=max_feat)

    # Here we fit the vectorized data to our k-Means algorithm:
    kmeans_clusterer = KMeans(n_clusters=opt_clusters,
                              n_jobs=-1,
                              verbose=1,
                              random_state=rand_state
                              )
    kmeans_clusterer.fit(X)

    # Here, we create our recommendation DF so that we can append recs to it:
    kmeans_recommendation_df = pd.DataFrame(columns=["series", "season", "episode", "ep_title", "identifier", "IMDB_user_rating"])

    # After the data is fit to the vectorizer, we can determine our cluster centers and their top 15 terms:
    centroid_array = kmeans_clusterer.cluster_centers_
    top_centroids = centroid_array.argsort()[:, -1:-16:-1]
    print("Top 15 words for each cluster:\n")
    for num, centroid in enumerate(top_centroids):
        print(f"Centroid {num} :",", ".join(features_arr[i] for i in centroid),"\n")

    # From here to the end of the function, we print out one not-so-random episode for each cluster:
    print(f"A random GOOD ({imdb_rating_threshhold}+ rating) episode in each cluster:\n")
    assigned_cluster = kmeans_clusterer.transform(X).argmin(axis=1)

    # These next 3 lines give us an episode rec proposal. If it passes the ratings screen in the next block of code, then it graduates from proposal to an actual recommendation:
    for idx in range(kmeans_clusterer.n_clusters):
        cluster = np.arange(0, X.shape[0])[assigned_cluster==idx]
        sample_ep_titles = np.random.choice(cluster, 1, replace=False)

        # This is where we filter out episodes below a certain IMDb ratings thresshold (as set by the parameter up top). The count variable is to guarantee that that we don't search forever in a cluster for an episode with a really high rating if (for example) no episodes in a given cluster has a rating that high:
        count = 0
        while (combined_master_df.loc[sample_ep_titles, "IMDB_user_rating"].values < imdb_rating_threshhold) and (count < (len(assigned_cluster) // 2) + 2):
            sample_ep_titles = np.random.choice(cluster, 1, replace=False)
            count += 1
            if combined_master_df.loc[sample_ep_titles, "IMDB_user_rating"].values >= imdb_rating_threshhold:
                continue

        # Prints cluster number prior to printing the random episode recommendation from that cluster:
        print("Cluster %d:" % idx)

        # Some text-prettying before printing the episode recommendation AND appending it to a recommendation DF:
        for ep_title in sample_ep_titles:
            if combined_master_df.loc[ep_title]["episode"] <= 9:
                episode = f'0{combined_master_df.loc[ep_title]["episode"]}'
            else:
                episode = f'{combined_master_df.loc[ep_title]["episode"]}'
            series = f'{combined_master_df.loc[ep_title]["series"]}'
            season = f'{combined_master_df.loc[ep_title]["season"]}'
            title = f'{combined_master_df.loc[ep_title]["ep_title"]}'
            ident = f'{combined_master_df.loc[ep_title]["identifier"]}'
            rating = f'{combined_master_df.loc[ep_title]["IMDB_user_rating"]}'
            print(f'{series} S:0{season} E:{episode} -- {title} -- {rating}')

            # This is where that aforementioned recommendation DF gets built:
            rec_row_list = [[series, int(season), int(episode), title, ident, float(rating)]]
            single_rec_df = pd.DataFrame(rec_row_list, columns=["series",
                                                                "season",
                                                                "episode",
                                                                "ep_title",
                                                                "identifier",
                                                                "IMDB_user_rating"])
            kmeans_recommendation_df = pd.concat([kmeans_recommendation_df, single_rec_df])

    return assigned_cluster, combined_master_df, kmeans_recommendation_df



def get_new_recs_from_existing_clusters(assigned_cluster, combined_master_df, imdb_rating_threshhold=7.0):
    """
    Function outputs new recommendations without having to rerun the clusterize function (which can be time-consuming).
    """

    # This is basically the same lines of code at the end of the clusterize function above. So long as you pass in the assigned_cluster variable and the combined_master_df as arguments, this function should be able to provide new recommendations:
    combined_master_df = create_df_for_vectorization()

    # Here, we create our recommendation DF so that we can append recs to it:
    kmeans_recommendation_df = pd.DataFrame(columns=["series", "season", "episode", "ep_title", "identifier", "IMDB_user_rating"])

    print(f"A random GOOD ({imdb_rating_threshhold}+ rating) episode in each cluster:\n")

    # These next 3 lines give us an episode rec proposal. If it passes the ratings screen in the next block of code, then it graduates from proposal to an actual recommendation:
    for idx in set(assigned_cluster):
        cluster = np.arange(0, assigned_cluster.shape[0])[assigned_cluster==idx]
        sample_ep_titles = np.random.choice(cluster, 1, replace=False)

        # This is where we filter out episodes below a certain IMDb ratings thresshold (as set by the parameter up top). The count variable is to guarantee that that we don't search forever in a cluster for an episode with a really high rating if (for example) no episodes in a given cluster has a rating that high:
        count = 0
        while (combined_master_df.loc[sample_ep_titles, "IMDB_user_rating"].values < imdb_rating_threshhold) and (count < (len(assigned_cluster) // 2) + 2):
            sample_ep_titles = np.random.choice(cluster, 1, replace=False)
            count += 1
            if combined_master_df.loc[sample_ep_titles, "IMDB_user_rating"].values >= imdb_rating_threshhold:
                continue

        # Prints cluster number prior to printing the random episode recommendation from that cluster:
        print("Cluster %d:" % idx)

        # Some text-prettying before printing the episode recommendation AND appending it to a recommendation DF:
        for ep_title in sample_ep_titles:
            if combined_master_df.loc[ep_title]["episode"] <= 9:
                episode = f'0{combined_master_df.loc[ep_title]["episode"]}'
            else:
                episode = f'{combined_master_df.loc[ep_title]["episode"]}'
            series = f'{combined_master_df.loc[ep_title]["series"]}'
            season = f'{combined_master_df.loc[ep_title]["season"]}'
            title = f'{combined_master_df.loc[ep_title]["ep_title"]}'
            ident = f'{combined_master_df.loc[ep_title]["identifier"]}'
            rating = f'{combined_master_df.loc[ep_title]["IMDB_user_rating"]}'
            print(f'{series} S:0{season} E:{episode} -- {title} -- {rating}')

            # This is where that aforementioned recommendation DF gets built:
            rec_row_list = [[series, int(season), int(episode), title, ident, float(rating)]]
            single_rec_df = pd.DataFrame(rec_row_list, columns=["series",
                                                                "season",
                                                                "episode",
                                                                "ep_title",
                                                                "identifier",
                                                                "IMDB_user_rating"])
            kmeans_recommendation_df = pd.concat([kmeans_recommendation_df, single_rec_df])

    return kmeans_recommendation_df



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



def vectorize(combined_master_df, max_features=None):
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
                                 tokenizer=tokenizer.tokenize,
                                 stop_words=stopwords_,
                                 max_features=None
                                 )

    # Here, we set our X to be the text we're looking to cluster; in this instance, the ultra_blob column of our DF. We want ALL of it:
    X = vectorizer.fit_transform(combined_master_df["ultra_blob"])

    features = vectorizer.get_feature_names()
    return X, features



def get_silhouette_score(max_clusters=24):
    """
    Determines the optimal number of clusters (default 24) and returns silhouette score graph. Some human judgment is going to be necessary to judge whether the highest scoring peaks are indeed the optimal number of max clusters, so some domain-specific knowledge would be helpful.
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
