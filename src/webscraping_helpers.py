import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

example_url = "https://www.imdb.com/title/tt0060028/episodes?season=1"

def scrape_season(target_url=example_url):
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


    show_season_ep = []
    for title, number in zip(range(len(title_strong)), ep_number_lst):
        show_season_ep.append(f"{soup.title.string[:-7]} " + "- " + f"{number}")
    title_lst = []
    for title in title_strong:
        title_lst.append(title.string)
    ep_number_lst = []
    for number in ep_no:
        ep_number_lst.append(number["content"])
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
