import numpy as np
import pandas as pd
import os

def create_combined_rev_rvr_info_df(DF_storage_directory="data/"):
    """
    Function to concatenate "reviews and reviewer" DFs for all series, and merge them with combined series info DF so that all reviews have series, season, and episode identifiers appended.

    NOTE: Must have combined_info_rev_df completed! Be sure to run combine_master_df.py first to create it!

    Output:
    ---
    *master_info_df, master_review_df: Pandas DataFrame with combined series/season/episode info and IMDB user review content, respectively
    """
    # Creating the DFs that we'll append the individual series info and user review DFs to:
    master_info_df = pd.read_pickle("data/combined_info_rev_df.pkl.bz2")

    data_dir = DF_storage_directory

    master_rev_rvr_df = pd.DataFrame(columns=['ep_title', 'rev_title', 'reviewer',
                                              'reviewer_link', 'rev_content',
                                              'rvr_rating'])

    for filename in os.listdir(data_dir):
        if filename.startswith("rvr_rev") and filename.endswith(".bz2"):
            rev_rvr_df = pd.read_pickle(f"{data_dir}{filename}")
            master_rev_rvr_df = pd.concat([master_rev_rvr_df, rev_rvr_df])

    # Clean up episode names by stripping out the series they're in (which is something that IMDB apparently does for completeness' sake):
    series_title_set = ['"Star Trek" ', '"Star Trek: Deep Space Nine" ',
                    '"Star Trek: Enterprise" ',
                    '"Star Trek: The Next Generation" ',
                    '"Star Trek: The Original Series" ',
                    '"Star Trek: Voyager" ']
    series_title_set = set(series_title_set)

    for idx, title in enumerate(master_rev_rvr_df["ep_title"].values):
        for series in series_title_set:
            if series in title:
                master_rev_rvr_df.loc[idx, "ep_title"] = title[len(series):]

    # Still cleaning up titles; here, we're standardizing them so that the titles all agree between the master IMDB info and reviews DF and the Memory Alpha summary DF:
    for idx, title in enumerate(master_rev_rvr_df["ep_title"].values):
        for series in series_title_set:
            if series in title:
                master_rev_rvr_df.loc[idx, "ep_title"] = title[len(series):]

    # Here, we take only what we need from the master_info_df and merge it to our master_rev_rvr_df:
    test_df = master_info_df[["ep_title", "show_sea_ep", "synopsis",
                              "IMDB_ID", "season", "episode", "series",
                              "year", "identifier"]]

    master_rev_revr_df = master_rev_revr_df.merge(test_df, on="ep_title")

    # Save the DF to a pickle object:
    master_rev_revr_df.to_pickle("data/combined_rev_rvr_df.pkl.bz2", compression="bz2")

    return master_rev_revr_df
