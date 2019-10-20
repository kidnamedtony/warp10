import numpy as np
import pandas as pd
import os

def create_combined_master_df(DF_storage_directory="data/"):
    """
    Output:
    ---
    *master_info_df, master_review_df: Pandas DataFrame with combined series/season/episode info and IMDB user review content, respectively
    """
    # Creating the DFs that we'll append the individual series info and user review DFs to:
    master_info_df = pd.DataFrame(columns=['show_sea_ep', 'ep_title', 'airdate',
                                           'synopsis', 'IMDB_ID', 'IMDB_user_rating',
                                           'num_IMDB_usr_rtgs', 'season', 'episode',
                                           'series', 'year'])
    master_review_df = pd.DataFrame(columns=["episode", "rev_title",
                                             "rev_content", "rvr_rating"])

    data_dir = DF_storage_directory

    for filename in os.listdir(data_dir):
        if filename.startswith("Star") and filename.endswith(".bz2"):
            info_df = pd.read_pickle(f"{data_dir}{filename}")
            master_info_df = pd.concat([master_info_df, info_df])
        elif filename.startswith("reviewsStar") and filename.endswith(".bz2"):
            rev_df = pd.read_pickle(f"{data_dir}{filename}")
            master_review_df = pd.concat([master_review_df, rev_df])
        else:
            continue

    # Making all the ratings numbers into numbers (rather than strings). Some reviewers didn't give their reviews a rating however, so we'll coerce them to NaNs:
    master_review_df["rvr_rating"] = pd.to_numeric( master_review_df["rvr_rating"], errors="coerce")

    # The above numeric coercion doesn't return an actual DF, so we'll need to make it one after we determine the average rating by users who did rate episodes along w/their reviews:
    master_rev_ep_df = master_review_df.groupby(["episode"])["rvr_rating"].mean()
    master_rev_ep_df = pd.DataFrame(master_rev_ep_df.copy())
    master_rev_ep_df = master_rev_ep_df.reset_index()

    # Here, we'll clean up and combine the different review titles for each episode's reviews by IMDB users before tying them them to their respective episode:
    ep_rev_dict = {}
    temp_title = []
    for ep, title in master_review_df.groupby(["episode"])["rev_title"]:
        ep_rev_dict[ep] = str(title.values).replace("[", "").replace("]","")

    for idx, title in enumerate(master_rev_ep_df["episode"].values):
        master_rev_ep_df.loc[idx, "title_blob"] = ep_rev_dict[title]

    # And we now do the same as the above step, but for the actual reviews themselves:
    ep_rev_cont_dict = {}
    temp_rev = []
    for ep, content in master_review_df.groupby(["episode"])["rev_content"]:
        ep_rev_cont_dict[ep] = str(content.values).replace("[", "").replace("]","")

    for idx, content in enumerate(master_rev_ep_df["episode"].values):
        master_rev_ep_df.loc[idx, "rev_blob"] = ep_rev_cont_dict[content]

    # This is a "Star Trek"-only step to clean up the episode titles in master_rev_ep_df:
    series_title_set = ['"Star Trek" ', '"Star Trek: Deep Space Nine" ', '"Star Trek: Enterprise" ', '"Star Trek: The Next Generation" ', '"Star Trek: The Original Series" ', '"Star Trek: Voyager" ']
    series_title_set = set(series_title_set)

    for idx, title in enumerate(master_rev_ep_df["episode"].values):
        master_rev_ep_df.loc[idx, "episode"] = title[:-18]

    for idx, title in enumerate(master_rev_ep_df["episode"].values):
        for series in series_title_set:
            if series in title:
                master_rev_ep_df.loc[idx, "episode"] = title[len(series):]

    # Now, we merge both DFs on their episode titles, which should finally match:
    master_info_df = master_info_df.set_index("ep_title")
    master_rev_ep_df = master_rev_ep_df.rename(columns={"episode": "ep_title"})

    combined_master_df = master_info_df.merge(master_rev_ep_df, on="ep_title")

    # Drop duplicate values (in our Star Trek info DFs, there are redundancies from scraping for some reason I haven't been able to track down quite yet. This should fix them, though):
    combined_master_df.drop_duplicates(inplace=True)
    combined_master_df = combined_master_df.reset_index().drop("index", axis=1)

    # Here we smash all the text columns into one (so that we can cluster it with KMeans later):
    combined_master_df["super_blob"] = combined_master_df["synopsis"]+" "+combined_master_df["title_blob"]+" "+combined_master_df["rev_blob"]
    combined_master_df.to_pickle("data/combined_info_rev_df.pkl.bz2", compression="bz2")

    # And finally, we give the DF an identifier column, smashing Series, Season, and Episode columns together:
    for idx, episode in enumerate(combined_test["episode"].values):
        if episode < 10:
            combined_test.loc[idx, "identifier"] = f"{combined_test.loc[idx, 'series']}" + "s0" + f"{combined_test.loc[idx, 'season']}" + "e0" + f"{combined_test.loc[idx, 'episode']}"
        else:
            combined_test.loc[idx, "identifier"] = f"{combined_test.loc[idx, 'series']}" + "s0" + f"{combined_test.loc[idx, 'season']}" + "e" + f"{combined_test.loc[idx, 'episode']}"

    return combined_master_df
