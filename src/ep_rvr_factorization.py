import numpy as np
import pandas as pd

from surprise import NMF, accuracy, Reader, Dataset
from surprise.model_selection import GridSearchCV, train_test_split

def matrix_factorize_for_latent_feats():
    """
    Function employs matrix factorization (NMF) of user scores for the episodes they've seen. Typically, this type of recommender algorithm aims to predict reviewer scores for episodes that users *haven't* seen, with user and episode latent features as a byproduct. This function will return a dataframe of episodes strongest in each latent feature instead.

    Inputs:
    ---
    None as yet. This code should just run once the function is called.

    Output:
    ---
    *matfact_lf_recs: Pandas DataFrame; DF containing the top episodes for each latent feature (as determined by the algorithm)
    """

    # First, we must build our reviewer DFs from which we will factorize user ratings:
    rev_rvr_df = pd.read_pickle("data/combined_rev_rvr_df.pkl.bz2")

    # Drop duplicate values in reviews and reviewer DF:
    rev_rvr_df.drop_duplicates(inplace=True)

    summary_df = pd.read_pickle("data/combined_info_rev_df.pkl.bz2")

    # Here, we create dictionaries for episode IDs and reviewer IDs to be applied to the recommendation DFs we'll be creating below:
    id_ep = dict(enumerate(rev_rvr_df["ep_title"].unique()))
    ep_id = dict((values, keys) for keys, values in id_ep.items())
    id_rvr = dict(enumerate(rev_rvr_df["reviewer"].unique()))
    rvr_id = dict((values, keys) for keys, values in id_rvr.items())

    # Append reviewer and episode IDs to reviews DF using the dictionaries created above:
    rev_rvr_df["rvr_id"] = None
    rev_rvr_df["ep_id"] = None
    rev_rvr_df.loc[:, 'ep_id'] = rev_rvr_df["ep_title"].map(lambda x: ep_id.get(x, np.nan))
    rev_rvr_df.loc[:, "rvr_id"] = rev_rvr_df["reviewer"].map(lambda x: rvr_id.get(x, np.nan))

    # Fill empty reviewer scores with mean values so we can create our utility matrix and so the NMF algo can work without throwing errors:
    rev_rvr_df["rvr_rating"].fillna(value=rev_rvr_df["rvr_rating"].mean(), inplace=True)

    # Creating reader object to read in our DF. This is a necessary step to create the Dataset object from our DF that the Surprise library likes to work with:
    reader = Reader(rating_scale=(1.0, 10.0))
    data = Dataset.load_from_df(rev_rvr_df[["rvr_id", "ep_id", "rvr_rating"]], reader=reader)

    # Splitting our data into training and testing sets for the algorithm:
    trainset, testset = train_test_split(data, test_size=0.2)

    # Creating and fitting our NMF algorithm with optimal hyperparameters (per gridsearching at a previous date):
    optimal_nmf = NMF(n_factors=20,
                      n_epochs=100,
                      reg_bi=0.15,
                      reg_qi=0.3,
                      lr_bu=0.005,
                      lr_bi=0.001,
                      biased=True,
                      verbose=False)
    optimal_nmf.fit(trainset)

    # Brief aside to check our error metrics, for those curious:
    predicted_nmf = optimal_nmf.test(testset)
    print(accuracy.mae(predicted_nmf))
    print(accuracy.rmse(predicted_nmf))

    # Now we create a DF for ALL episodes and their latent feature scores in the columns:
    latent_feat = pd.DataFrame(optimal_nmf.qi)

    # Append episode titles to the big latent_feat DF using the id_ep dictionary we created above (as a reminder: that's the dictionary where keys are episode indices in the big latent_feat DF and vals are episode titles):
    latent_feat["ep_title"] = pd.Series(id_ep)

    # Creating a list of episodes that have the highest value for each latent feature so that we can create a separate DF of just these episodes:
    max_lf_vals = list(latent_feat.iloc[:, :-1].max())

    # Here, we create a new DF to contain ONLY the episodes that are highest in each latent feature:
    max_lf_df = pd.DataFrame(columns=latent_feat.columns)

    for col_idx, vals in enumerate(max_lf_vals):
        temp_df = pd.DataFrame(latent_feat[latent_feat.loc[:, col_idx] == vals])
        max_lf_df = pd.concat([max_lf_df, temp_df])

    # Appending series-season-episode identifiers, IMDb user rating, and episode titles from the summary_df by merging on episode titles (now that both have ep titles):
    max_lf_df = max_lf_df.merge(summary_df[["series",
                                            "season",
                                            "episode",
                                            "ep_title",
                                            "identifier",
                                            "IMDB_user_rating"]],
                                on="ep_title")
    matfact_lf_recs = max_lf_df.loc[:, ["series", "season", "episode", "ep_title", "identifier", "IMDB_user_rating"]]

    return matfact_lf_recs

# def matrix_factorize(num_of_components=13):
#     """
#     Factorize a ratings dataframe to find latent features. Returns a DF of episodes as rows and latent features as columns.
#
#     Input:
#     ---
#     *num_of_components: int; how many latent features you want to find.
#
#     Output:
#     ---
#     *Q_df: Pandas DataFrame; DF of episodes as rows and latent features as columns.
#     """
#     # Pull in reviews and reviewer DataFrame (combined_rev_rvr_df):
#     combined_rev_revr_df = pd.read_pickle("data/combined_rev_rvr_df.pkl.bz2")
#
#     # Fill NaNs w/mean values in the combined_rev_rvr_df so that we can pivot it and make a utility matrix:
#     combined_rev_revr_df["rvr_rating"].fillna(value=combined_rev_revr_df["rvr_rating"].mean(), inplace=True)
#
#     utility = pd.pivot_table(combined_rev_revr_df,
#                              values="rvr_rating",
#                              index="reviewer",
#                              columns="ep_title")
#
#     # Create ratings_df:
#     reviewers = list(combined_rev_revr_df["reviewer"].unique())
#     episodes = list(combined_rev_revr_df["ep_title"].unique())
#
#     ratings_df = pd.DataFrame(index=reviewers, columns=episodes)
#     ratings_df = utility.loc[users, :]
#
#     # Fill NaNs in the ratings_df (as inherited from the utility matrix) with mean values so that SVD can work with it:
#     ratings_df = utility.fillna(value=utility.mean())
#
#     # Instantiate SVD algorithm and fit ratings_df to it:
#     svd = TruncatedSVD(n_components=num_of_components)
#     svd.fit(ratings_df)
#
#     # Create P and Q matrices. P = reviewers vs. latent features, Q = episodes vs. latent features
#     P = svd.transform(ratings_df)
#     Q = svd.components_.T
#
#     # Print RMSE of the algorithm:
#     print(f"RMSE:{((ratings_df.values - P @ Q.T)**2).mean()**0.5}")
#
#     # Put Q into a DF with labeled columns and rows, to make things easier to look at and understand
#     Q_df = pd.DataFrame(Q, index=ratings_df.columns)
#     display(Q_df)
#
#     return Q_df
