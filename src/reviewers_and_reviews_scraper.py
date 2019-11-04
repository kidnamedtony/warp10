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

# For Star Trek Only!
trek_series_links_lst = ["https://www.imdb.com/title/tt0060028/",
                         "https://www.imdb.com/title/tt0069637/",
                         "https://www.imdb.com/title/tt0092455/",
                         "https://www.imdb.com/title/tt0106145/",
                         "https://www.imdb.com/title/tt0112178/",
                         "https://www.imdb.com/title/tt0244365/"
                         ]

def scrape_all_reviews(target_url_list, create_master_df=False):
    """
    All in one function that, should the user provide a list of series links, scrape episode reviews from all of them. Calls on the scrape_series_reviews function, below.

    Input:
    ---
    *target_url_list: List; list of string of the series page URLs that we want to scrape reviews from.
    *creat_summary_master_df: Boolean; usually set to false. Will create a Master DF to append reviews to if True.

    Output:
    ---
    Pretty much the same as scrape_series_reviews function (a dataframe and a pickled object), but only as often as the links in the list it's first fed.
    """
    for link in target_url_list:
        scrape_series_reviews(link, create_master_df)


# By way of example, here's the IMDB page for "Star Trek: The Original Series":
example_main_target_url = "https://www.imdb.com/title/tt0060028/"

def scrape_series_reviews(target_url, create_reviews_master_df=False):
    """
    All-in-one function to scrape IMDB info for all reviews of a given show, per episode, and put them into a Pandas DataFrame. Calls on modified versions of scrape_season, get_and_clean_details, and add_season functions once season data is scraped.

    Input:
    ---
    *target_url: String; contains URL of the show we'll be scraping from IMDB. (Default is set to Star Trek: The Original Series.)
    * create_reviews_master_df: Boolean; will create a master dataframe for scraped data to be added to. (Best used if running for this script for the first time. Thus, default is set to False.)

    Output:
    ---
    A Pandas DF
    """
    # If running for the first time and create_series_master_df is set to True, this will create the master DF
    if create_reviews_master_df == True:
        reviews_master_df = pd.DataFrame(columns=["ep_title", "rev_title", "reviewer", "reviewer_link", "rev_content", "rvr_rating"])

    print(f"URL of the show we'll be scraping: {target_url}")
    logger.info(f"URL of the show we'll be scraping: {target_url}")

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

    # Grabbing seasons and years links (because they're in the same element) from target_url:
    seasons_yrs_div = soup.find("div", {"class": "seasons-and-year-nav"})

    # Creating a list of links for each season to scrape further information:
    seasons_link_lst = []
    for lnk in seasons_yrs_div.find_all("a"):
        if "year" not in lnk["href"]:
            seasons_link_lst.append(f"https://www.imdb.com{lnk['href']}")
    logger.info(f"Created seasons_link_lst:\n {seasons_link_lst} \n")

    # Set up request monitor for scraping season data
    start_time = time.time()
    request = 0
    logger.info("Starting timer for season info scraping")

    # creating an empty list for episode links to be stored outside of the following loop so that episodes from ALL seasons can be stored here:
    ep_link_lst = []
    print(f"Ep link list after creation: {ep_link_lst}")

    # This is where we start scraping details for each season of our TV show:
    for season in seasons_link_lst:
        print(f"Season we're scraping: {season}")

        # Pause for a bit, to simulate clicking through pages at human speeds:
        sleep(randint(3,5))

        # Make a request for a season:
        logger.info(f"Scraping season data for {season}")
        season_response = requests.get(season)
        print(f"Status code: {season_response.status_code}")

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
        if request > 10:
            warn("Number of requests greater than expected (10); Breaking loop")
            logger.warning("Number of requests greater than expected (10); Breaking loop")
            break

        # Build soup object for season of show, after which we'll pull links for individual episodes
        season_soup = BeautifulSoup(season_response.content, "html.parser")
        logger.info("Built season soup object")

        # Now we get and build a set of links for each episode:
        all_eps_div = season_soup.find("div", {"class": "list detail eplist"})
        ep_links = all_eps_div.find_all("strong")

        for link in ep_links:
            ep_link_lst.append(f"https://www.imdb.com{link.find('a')['href']}")
        print(f"Ep link list after appending ep links: {ep_link_lst}")

    for episode in ep_link_lst:
        print(f"Episode we're scraping: {episode}")

        sleep(randint(3,5))

        # Request data from the episode in question's page on IMDB:
        ep_response = requests.get(episode)
        logger.info(f"Scraping season data for {episode}, status: {ep_response.status_code}")
        print(f"Episode page status code: {ep_response.status_code}")

        # This is how we monitor requests as they happen AND add to the count of requests:
        request += 1
        print(f"Request:{request}; Freq: {request/elapsed_time}; request/sec")
        logger.info(f"Request:{request}; Freq: {request/elapsed_time}; request/sec")

        # Checks to see if scraping goes okay. 200 means all systems are go!
        if season_response.status_code != 200:
            warn(f"Request: {request}; Status code: {season_response.status_code}")
            logger.info(f"Request: {request}; Status code: {season_response.status_code}")

        # Turn page response of series into a soup object and then find the "view all reviews" link, so we can get to the page with all the user reviews:
        ep_soup = BeautifulSoup(ep_response.content, "html.parser")
        top_review = ep_soup.find("div", {"id": "titleUserReviewsTeaser"})
        all_reviews_link = f"https://www.imdb.com{top_review.find_all('a')[-1]['href']}"

        # Grab episode title:
        ep_title = ep_soup.title.text[:-7]
        print(f"Scraping reviews for: {ep_title}")

        # Wait for a few seconds before our next request:
        sleep(randint(3,5))

        # Request all the episode's reviews:
        review_page_response = requests.get(all_reviews_link)
        logger.info(f"Reviews page we're scraping: {all_reviews_link}, status code: {review_page_response.status_code}")
        print(f"Reviews page we're scraping: {all_reviews_link}, status code: {review_page_response.status_code}")

        # Turn review page into soup object:
        rev_pg_soup = BeautifulSoup(review_page_response.content, "html.parser")

        # Extract all review, reviewers, scores, etc. from review page soup:
        review_containers = rev_pg_soup.find_all("div", {"class": "review-container"})

        # From review_containers, take each user's name, review, score, account page, etc. for the episode they're reviewing and store all this in a list for throwing into a DF in the next step:
        print(f"Assembling reviews DF for: {ep_title}")
        ep_rev = []
        for idx, entry in enumerate(review_containers):
            temp_title = ""
            temp_reviewer = ""
            temp_reviewer_link = ""
            temp_content = ""
            temp_rating = ""
            for title in (review_containers[idx].find_all("a", {"class": "title"})):
                temp_title += (title.text.lstrip().rstrip())
            for content in (review_containers[idx].find_all("div", {"class": "text show-more__control"})):
                temp_content += (content.text.lstrip().rstrip())
            for rating in (review_containers[idx].find_all("span", {"class": "rating-other-user-rating"})):
                temp_rating += (rating.text[:-4].lstrip().rstrip())
            for name in (review_containers[idx].find_all("span", {"class": "display-name-link"})[0].find("a").text):
                temp_reviewer += (name.lstrip().rstrip())
            temp_reviewer_link += (f'https://www.imdb.com{review_containers[idx].find_all("span", {"class": "display-name-link"})[0].find("a")["href"]}')
            ep_rev.append([ep_title[:-18], temp_title, temp_reviewer, temp_reviewer_link, temp_content, temp_rating])

        # Append all the review content and info to a temporary DF before appending it to the series review master DF:
        temp_ep_rev_df = pd.DataFrame(ep_rev, columns=["ep_title", "rev_title", "reviewer", "reviewer_link", "rev_content", "rvr_rating"])
        print("Shape of temp reviews DF:", temp_ep_rev_df.shape)

        # Turn our user ratings into floats instead of strings, and coerce those that didn't leave a rating into NaN values:
        temp_ep_rev_df["rvr_rating"] = pd.to_numeric(temp_ep_rev_df["rvr_rating"], errors="coerce")

        # Append the reviews from the temporary DF to the master DF
        reviews_master_df = pd.concat([reviews_master_df, temp_ep_rev_df])
        print("Shape of Master DF:", reviews_master_df.shape)

    # Save master DFs as pickle object AND CSV:
    reviews_master_df.reset_index(drop="index", inplace=True)

    reviews_master_df.to_pickle(f"data/rvr_rev{soup.title.text.replace(' ', '-')[:-7]}.pkl.bz2", compression="bz2")
    logger.info("Saved Reviews Master DF to pickle object")

    reviews_master_df.to_csv(f"data/rvr_rev{soup.title.text.replace(' ', '-')[:-7]}.csv")
    logger.info("Saved Reviews Master DF to CSV")

    return reviews_master_df
