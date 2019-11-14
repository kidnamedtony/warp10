import random
import numpy as np
import pandas as pd
import os

def hybridize_and_get_recs(kmeans_recommendation_df, matfact_lf_recs, num_eps_to_rec=6, imdb_ratings_threshhold=7.0):
    """
    Function to take recommendations from all algorithms and return hybridized results based on however many episodes the user wants to watch.
    """
    # First, we combine the recommendation dfs from both our kmeans cluster algo and the NMF algo. We reset the index as well, for future steps:
    combined_recs = pd.concat([kmeans_recommendation_df, matfact_lf_recs]).reset_index(drop="index")

    # Next, we get random indices from the DF for as many episodes as the user wants.
    shuffled_idx = random.sample(set(combined_recs.index), k=num_eps_to_rec)

    # Finally, we output a selection of recommendations randomly chosen from the combined_recs DF:
    for idx in shuffled_idx:
        display(combined_recs.loc[idx, :])
