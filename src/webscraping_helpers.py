import numpy as np
import pandas as pd
import requests
import time
from time import sleep
from random import randint
from warnings import warn
from bs4 import BeautifulSoup

from . import custom_logger as CL

logger = CL.logger

example_main_target_url = "https://www.imdb.com/title/tt0060028/"

def scrape_series_details(target_url, create_series_master_df=False):
    """
    All-in-one function to scrape IMDB info for all seasons of a given show and put them into a Pandas DataFrame. Calls on scrape_season, get_and_clean_details, and add_season functions once season data is scraped.

    Input:
    ---
    *target_url: String; contains URL of the show we'll be scraping from IMDB. (Default is set to Star Trek: The Original Series.)
    * create_series_master_df: Boolean; will create a master dataframe for scraped data to be added to. (Best used if running for this script for the first time. Thus, default is set to False.)

    Output:
    ---
    A Pandas DF
    """
    print("URL of the show we'll be scraping:", target_url)
    logger.info("URL of the show we'll be scraping:", target_url)

    # Make a request for one particular TV show (default is ST: TOS):
    series_response = requests.get(target_url)
    logger.info(f"Scraping for list of seasons from {target_url}")

    # Checks to see if scraping goes okay. 200 means all systems are go!
    if series_response.status_code != 200:
        warn(f"Warning: status code {series_response.status_code}")
        logger.warning(f"Warning: status code {series_response.status_code}")
    else:
        print("Status code 200; all good in the hood")
        logger.info(f"Status code: {series_response.status_code}")

    # If the status code is all good, we request on!
    soup = BeautifulSoup(series_response.content, "html.parser")
    logger.info(f"First 500 chars of soup object:\n {soup.text[:500]} \n")

    # Grabbing seasons and years links (because they're in the same element) from target_url:
    seasons_yrs_div = soup.find("div", {"class": "seasons-and-year-nav"})
    logger.info(f"seasons_yrs_div:\n {seasons_yrs_div} \n")

    # Creating a list of links for each season to scrape further information:
    seasons_link_lst = []
    temp_lst = []
    for num, links in enumerate(seasons_yrs_div.find_all("a")):
        temp_lst.append(seasons_yrs_div.find_all("a")[num]["href"])
    for lnk in temp_lst[::-1]:
        if "year" not in lnk:
            seasons_link_lst.append("https://www.imdb.com" + lnk)
    logger.info(f"Created seasons_link_lst:\n {seasons_link_lst} \n")

    # If running for the first time and create_series_master_df is set to True, this will create the master DF
    if create_series_master_df == True:
        series_master_df = pd.DataFrame()
        logger.info("Created Master DataFrame")

    # Set up request monitor for scraping season data
    start_time = time.time()
    request = 0
    logger.info("Starting timer for season info scraping")

    # This is where we start scraping details for each season of our TV show:
    for season in seasons_link_lst:
        print(season)

        # Pause for a bit, to simulate clicking through pages at human speeds:
        sleep(randint(9,18))

        # Make a request for a season:
        logger.info(f"Scraping season data for {season}")
        season_response = requests.get(season)

        # This is how we monitor requests as they happen AND add to the count of requests:
        request += 1
        elapsed_time = time.time() - start_time
        print(f"Request:{request}; Freq: {request/elapsed_time}; request/sec")
        logger.info(f"Request:{request}; Freq: {request/elapsed_time}; request/sec")

        # Checks to see if scraping goes okay. 200 means all systems are go!
        if season_response.status_code != 200:
            warn(f"Request: {request}; Status code: {season_response.status_code}")
            logger.info(f"Request: {request}; Status code: {season_response.status_code}")

        # Break loop if it goes over an unreasonable amount of requests:
        if request > 8:
            warn("Number of requests greater than expected (10); Breaking loop")
            logger.warning("Number of requests greater than expected (10); Breaking loop")
            break

        season_soup = scrape_season(season)
        logger.info("Built season soup object")

        season_df = get_and_clean_details(season_soup)
        logger.info("Built DF for season")

        series_master_df = add_seasons(series_master_df, season_df)
        logger.info("Added season DF to master DF")

    # Saving Series Master DF to pickle object:
    #series_master_df.to_pickle("data/series_master_df.pkl.bz2", compression="bz2")
    logger.info("Saved Series Master DF to pickle object")

    # Saving Series Master DF data to (optional) CSV:
    series_master_df.to_csv(f"data/{soup.title.string.replace(' ', '-')}.csv")
    logger.info("Save Series Master DF data to CSV")

    return series_master_df


"""
Find way to not repeat work (e.g. check dataframe to see if show was scraped already, in case of crashes)
"""

example_season_url = "https://www.imdb.com/title/tt0060028/episodes?season=1"

def scrape_season(target_url=example_season_url):
    """
    Makes BeautifulSoup object of scraped TV show data (per given season of the show that the input URL points to).

    Input:
    ---
    *target_url: string containing the URL of the TV show season/series. Default URL is the 1st season of Star Trek: The Original Series

    Output:
    ---
    *soup: BeautifulSoup object
    """
    response = requests.get(target_url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup

def get_and_clean_details(soup):
    """
    Takes content from soup object (like those created by scrape_season function) and cleans up all the interesting data before placing them into a Pandas DataFrame.

    Input:
    ---
    *soup: BeautifulSoup object (like those created by scrape_season function)

    Output:
    ---
    *df: Pandas DataFrame of cleaned up TV show data
    """
    all_eps_div = soup.find("div", {"class": "list detail eplist"})
    title_strong = all_eps_div.find_all("strong")
    ep_no = all_eps_div.find_all("meta")
    airdate_div = all_eps_div.find_all("div", {"class": "airdate"})
    desc_div = all_eps_div.find_all("div", {"class": "item_description"})
    rating_span = all_eps_div.find_all("span", {"class": "ipl-rating-star__rating"})
    no_usr_rtgs_span = all_eps_div.find_all("span", {"class": "ipl-rating-star__total-votes"})


    ep_number_lst = []
    for number in ep_no:
        ep_number_lst.append(number["content"])
    title_lst = []
    for title in title_strong:
        title_lst.append(title.string)
    ep_id_lst = []
    for link in title_strong:
        ep_id_lst.append("https://www.imdb.com/title/" + link.find("a")["href"][7:-1])
    airdate_lst = []
    for date in airdate_div:
        airdate_lst.append(date.string.lstrip().rstrip())
    desc_lst = []
    for desc in desc_div:
        desc_lst.append(desc.string.lstrip().rstrip())
    rating_lst = []
    for rating in rating_span[::23]:
        rating_lst.append(rating.string)
    no_usr_rtgs_lst = []
    for number in no_usr_rtgs_span:
        no_usr_rtgs_lst.append(int(number.string[1:-1].replace(",", "")))
    show_season_ep = []
    for title, number in zip(range(len(title_strong)), ep_number_lst):
            show_season_ep.append(f"{soup.title.string[:-7]} " + "- " + f"{number}")

    combined_dict = {
        "show_sea_ep": show_season_ep,
        "ep_title": title_lst,
        "airdate": airdate_lst,
        "synopsis": desc_lst,
        "IMDB_ID": ep_id_lst,
        "IMDB_user_rating": rating_lst,
        "num_IMDB_usr_rtgs": no_usr_rtgs_lst
        }
    df = pd.DataFrame(combined_dict)

    return df

"""
NOTE TO SELF: Eventually, we're going to want to rewrite this script such that it saves the new DFs to a list that it can later process in one fell swoop into a combined DF. All we feed it, ostensibly, is the link to the show.
"""

def add_seasons(master_df, df_to_add):
    """
    WIP: Function to combine TV show season DataFrames.

    Input:
    ---
    *master_df: Pandas DF; the DF we'll be adding all future season and series DFs to.
    *df_to_add: Pandas DF; the DF we'll be adding to the master DF.

    Output:
    ---
    master_df: Pandas DF, with added content.
    """
    master_df = pd.concat([master_df, df_to_add])
    master_df = master_df.reset_index().drop("index", axis=1)
    return master_df
