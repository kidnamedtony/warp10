import numpy as np
import pandas as pd
import os
import unidecode

"""
Note: You may need to install Unidecode (pip install Unidecode) for a part of this script to work.
"""

def combine_rev_rvr_and_summary_to_info_dfs(DF_storage_directory="data/"):
    """
    Function to concatenate "reviews and reviewer" DFs for all series, and merge them with combined series info DF so that all reviews have series, season, and episode identifiers appended.

    NOTE: Must have combined_info_rev_df completed! Be sure to run combine_master_df.py first to create it!

    Output:
    ---
    *master_info_df, master_review_df: Pandas DataFrame with combined series/season/episode info and IMDB user review content, respectively
    """
    # Creating the rev_rvr and wikia summary DFs that we'll append the individual series info DFs to:
    master_info_df = pd.read_pickle("data/combined_info_rev_df.pkl.bz2")

    data_dir = DF_storage_directory

    master_rev_rvr_df = pd.DataFrame(columns=['ep_title', 'rev_title', 'reviewer',
                                              'reviewer_link', 'rev_content',
                                              'rvr_rating'])
    master_summary_df = pd.DataFrame(columns=['ep_title', 'summary'])

    for filename in os.listdir(data_dir):
        if filename.startswith("rvr_rev") and filename.endswith(".bz2"):
            rev_rvr_df = pd.read_pickle(f"{data_dir}{filename}")
            master_rev_rvr_df = pd.concat([master_rev_rvr_df, rev_rvr_df])

    for filename in os.listdir(data_dir):
    if filename.startswith("summary") and filename.endswith(".bz2"):
        summary_df = pd.read_pickle(f"{data_dir}{filename}")
        master_summary_df = pd.concat([master_summary_df, summary_df])

    # Here's a bit of Memory Alpha wikia weirdness: we have to append three additional entries that the wiki doesn't account for as they're two-parters lumped together with their counterpart ep:
    part_twos_df = pd.DataFrame(columns=["ep_title", "summary"])

    part_twos_df = pd.concat([part_twos_df, master_summary_df[master_summary_df["ep_title"] == "Broken Bow"]])
    part_twos_df = pd.concat([part_twos_df, master_summary_df[master_summary_df["ep_title"] == "Flesh and Blood"]])
    part_twos_df = pd.concat([part_twos_df, master_summary_df[master_summary_df["ep_title"] == "Dark Frontier"]])
    part_twos_df.reset_index(drop="index", inplace=True)

    # Here, we rename them to "Part II", so that their second part now exists (and can be merged upon later, when we bring in the info DF):
    part_twos_df.loc[0, "ep_title"] = "Broken Bow: Part II"
    part_twos_df.loc[1, "ep_title"] = "Flesh and Blood: Part II"
    part_twos_df.loc[2, "ep_title"] = "Dark Frontier: Part II"

    # And we append "Part I" to a few other first-part episodes, in case they don't already have that in their title:
    master_summary_df.loc[451, "ep_title"] = "Broken Bow: Part I"
    master_summary_df.loc[655, "ep_title"] = "Dark Frontier: Part I"
    master_summary_df.loc[665, "ep_title"] = "Equinox: Part I"

    # Reset indices:
    master_rev_revr_df.reset_index(drop="index", inplace=True)
    master_summary_df.reset_index(drop="index", inplace=True)

    # Clean up episode names by stripping out the series they're in (which is something that IMDB apparently does for completeness' sake--this is only for IMDB scraped content):
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

    # Here, we take only what we need from the master_info_df before merging it to our master_rev_rvr_df:
    test_df = master_info_df[["ep_title", "show_sea_ep", "synopsis",
                              "IMDB_ID", "season", "episode", "series",
                              "year", "identifier"]]

    # But wait, there's even more title weirdness! This time in the summary DF. Here's how we fix it now that we've created that test_df:

    # First, create a list of titles not in the test_df:
    ep_titles_not_in_test = []
    for idx, sum_title in enumerate(master_summary_df["ep_title"].values):
        if sum_title not in test_df["ep_title"].values:
            print(idx, sum_title)
            ep_titles_not_in_test.append((idx, sum_title))

    # Then we do the same for the summary DF:
    ep_titles_not_in_summary_df = []
    for idx, test_title in enumerate(test_df["ep_title"].values):
        if test_title not in master_summary_df["ep_title"].values:
            print(idx, test_title)
            ep_titles_not_in_summary_df.append((idx, test_title))

    # And then, we place overwrite the titles in the summary DF with the titles in test_df, so that we have them nice and uniform:
    new_titles_for_summary_df = []
    for idx, title in ep_titles_not_in_summary_df:
        for i, s_title in ep_titles_not_in_test:
            if unidecode.unidecode(title.lower()) not in unidecode.unidecode(s_title.lower()):
                continue
            else:
                new_titles_for_summary_df.append((i, title))

    # There's some weirdness with the above code block, as it misses three episode titles. I don't know why they weren't included, but oh well...I guess we can hard-code them for now (until I find out how to account for them above):
    new_titles_for_summary_df.append((224, 'Operation - Annihilate!'))
    new_titles_for_summary_df.append((121, '...Nor the Battle to the Strong'))
    new_titles_for_summary_df.append((150, 'You Are Cordially Invited...'))

    # This is where we overwrite the summary titles with the titles in test_df:
    for idx, title in new_titles_for_summary_df:
        master_summary_df.loc[idx, "ep_title"] = title

    # AND FINALLY, we merge our rev_revr and summary DFs with the details in test_df:
    master_rev_revr_df = master_rev_revr_df.merge(test_df, on="ep_title")
    master_summary_df = master_summary_df.merge(test_df, on="ep_title")

    # Save the DF to a pickle object:
    master_rev_revr_df.to_pickle("data/combined_rev_rvr_df.pkl.bz2", compression="bz2")
    master_summary_df.to_pickle("data/combined_summary_df.pkl.bz2", compression="bz2")

    return master_rev_revr_df, master_summary_df
