import os
import requests
import numpy as np
import pandas as pd
import time
from time import sleep
from random import randint
from warnings import warn
from bs4 import BeautifulSoup

from . import custom_logger as CL

logger = CL.logger

# Memory Alpha series links (obviously for Star Trek only):
series_links = ["https://memory-alpha.fandom.com/wiki/Star_Trek:_The_Original_Series",
                "https://memory-alpha.fandom.com/wiki/Star_Trek:_The_Animated_Series",
                "https://memory-alpha.fandom.com/wiki/Star_Trek:_The_Next_Generation",
                "https://memory-alpha.fandom.com/wiki/Star_Trek:_Deep_Space_Nine",
                "https://memory-alpha.fandom.com/wiki/Star_Trek:_Voyager",
                "https://memory-alpha.fandom.com/wiki/Star_Trek:_Enterprise"]

def scrape_all_summaries(target_url_list, create_master_df=False):
    """
    All in one function that, should the user provide a list of series links, scrape episode summaries from all of them. Calls on the scrape_series_summaries function, below.

    Input:
    ---
    *target_url_list: List; list of string of the series page URLs that we want to scrape summaries from.
    *creat_summary_master_df: Boolean; usually set to false. Will create a Master DF to append summaries to if True.

    Output:
    ---
    Pretty much the same as scrape_series_summaries function (a dataframe and a pickled object), but only as often as the links in the list it's first fed.
    """
    for link in target_url_list:
        scrape_series_summaries(link, create_master_df)

def scrape_series_summaries(target_url, create_summary_master_df=False):
    """
    Function to scrape Wikia fan wiki for episode summaries. So far, works best on Memory Alpha, the Star Trek wiki.

    Input:
    ---
    *target_url: String; string of the series page URL, which should have tables of all the episodes in all the seasons of the series.
    *creat_summary_master_df: Boolean; usually set to false. Will create a Master DF to append summaries to if True.

    Output:
    ---
    summary_master_df: Pandas DataFrame of episode titles and their respective plot summaries. Function also pickles this dataframe and stores it in the data/ folder.
    """
    if create_summary_master_df == True:
        summary_master_df = pd.DataFrame(columns=["ep_title", "summary"])

    print("URL of the show we'll be scraping:", target_url)
    logger.info(f"URL of the show we'll be scraping: {target_url}")

    # Make a request for the series in question, so we can grab links to each of its episodes:
    series_response = requests.get(target_url)
    logger.info(f"Scraping episode list from {target_url}")

    # Checks to see if scraping goes okay. 200 means all systems are go!
    if series_response.status_code != 200:
        warn(f"Warning: status code {series_response.status_code}")
        logger.warning(f"Warning: status code {series_response.status_code}")
    else:
        print("Status code 200; all good in the hood")
        logger.info(f"Status code: {series_response.status_code}")

    # If the status code is all good, we request on!
    series_soup = BeautifulSoup(series_response.content, "html.parser")

    # Pulling out all the td tags, where our episode links live:
    td = series_soup.find_all("td")

    # Storing episode links to a list (episode links occur every 6 entries):
    series_link_lst = []
    for idx, title in enumerate(td):
        if title.a != None:
            if "(episode)" in title.a["href"]:
                series_link_lst.append(f'https://memory-alpha.fandom.com{title.a["href"]}')

    # And from here, we start scraping episode summary content from our list of links:
    for link in series_link_lst:

        ep_target = link

        # Sleep for a moment, so that we don't get IP banned:
        sleep(randint(3,5))

        ep_resp = requests.get(ep_target)

        # Checks to see if scraping goes okay. 200 means all systems are go!
        if ep_resp.status_code != 200:
            warn(f"Warning: status code {ep_resp.status_code}")
            logger.warning(f"Warning: status code {ep_resp.status_code}")
        else:
            print("Status code 200; all good in the hood")
            logger.info(f"Status code: {ep_resp.status_code}")
        logger.info(f"Scraping episode summary from {ep_target}")

        # Creating our episode soup object:
        ep_soup = BeautifulSoup(ep_resp.content, "html.parser")

        # Stripping away the extraneous junk from the page's episode title and then saving it for later:
        ep_title_until_idx = ep_soup.title.text.index("(episode)")
        ep_title = ep_soup.title.text[:ep_title_until_idx].rstrip()

        # Saving our episode content div:
        ep_content_div = ep_soup.find("div", {"class": "mw-content-ltr mw-content-text"})

        # Setting the indices from and to the point that we want to capture in the next step (e.g., we want all text from "Summary" until "Memorable Quotes" on the page):
        from_summary = ep_content_div.text.index("Summary")
        until_memorable_quotes = ep_content_div.text.index("Memorable quotes")

        # The aforementioned "next step" where we save title and summary to a dictionary:
        summary_content = {"ep_title": [ep_soup.title.text[:ep_soup.title.text.index("(episode)")].rstrip()],
                   "summary": [ep_content_div.text[from_summary+13:until_memorable_quotes].lstrip()]}

        # Turning dictionary into a DF, and then appending it to the master DF:
        ep_summary_df = pd.DataFrame.from_dict(summary_content)
        summary_master_df = pd.concat([summary_master_df, ep_summary_df])
        print("Shape of Master DF:", summary_master_df.shape)

    # Once the episode summary scraping loop completes, we pickle our master DF:
    title_until = series_soup.title.text.index(" |")
    summary_master_df.to_pickle(f"data/summary{series_soup.title.text[:title_until]}.pkl.bz2", compression="bz2")
    logger.info("Saved Summary Master DF to pickle object")

    # And we save it as a CSV too:
    summary_master_df.to_csv(f"data/summary{series_soup.title.text[:title_until]}.csv")
    logger.info("Saved Summary Master DF to CSV")

    return summary_master_df
