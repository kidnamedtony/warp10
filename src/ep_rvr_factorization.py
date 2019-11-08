import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from surprise import SVD

def matrix_factorize(num_of_components=13):
    """
    Factorize a ratings dataframe to find latent features. Returns a DF of episodes as rows and latent features as columns.

    Input:
    ---
    *num_of_components: int; how many latent features you want to find.

    Output:
    ---
    *Q_df: Pandas DataFrame; DF of episodes as rows and latent features as columns.
    """
    # Pull in reviews and reviewer DataFrame (combined_rev_rvr_df):
    combined_rev_revr_df = pd.read_pickle("data/combined_rev_rvr_df.pkl.bz2")

    # Fill NaNs w/mean values in the combined_rev_rvr_df so that we can pivot it and make a utility matrix:
    combined_rev_revr_df["rvr_rating"].fillna(value=combined_rev_revr_df["rvr_rating"].mean(), inplace=True)

    utility = pd.pivot_table(combined_rev_revr_df,
                             values="rvr_rating",
                             index="reviewer",
                             columns="ep_title")

    # Create ratings_df:
    reviewers = list(combined_rev_revr_df["reviewer"].unique())
    episodes = list(combined_rev_revr_df["ep_title"].unique())

    ratings_df = pd.DataFrame(index=reviewers, columns=episodes)
    ratings_df = utility.loc[users, :]

    # Fill NaNs in the ratings_df (as inherited from the utility matrix) with mean values so that SVD can work with it:
    ratings_df = utility.fillna(value=utility.mean())

    # Instantiate SVD algorithm and fit ratings_df to it:
    svd = TruncatedSVD(n_components=num_of_components)
    svd.fit(ratings_df)

    # Create P and Q matrices. P = reviewers vs. latent features, Q = episodes vs. latent features
    P = svd.transform(ratings_df)
    Q = svd.components_.T

    # Print RMSE of the algorithm:
    print(f"RMSE:{((ratings_df.values - P @ Q.T)**2).mean()**0.5}")

    # Put Q into a DF with labeled columns and rows, to make things easier to look at and understand
    Q_df = pd.DataFrame(Q, index=ratings_df.columns)
    display(Q_df)

    return Q_df
