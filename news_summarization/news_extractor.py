"""
This module defines a class `NewsExtractor` for extracting news items
from sitemaps based on a given date.
"""
from time import sleep
from datetime import datetime
import logging
import re

from bs4 import BeautifulSoup
import trafilatura
import requests

log = logging.getLogger(__name__)


class NewsExtractor:  # pylint: disable=too-many-instance-attributes
    """
    A utility class for extracting news items from website sitemaps based on a given date.

    This class facilitates the retrieval of sitemaps from a website's robots.txt file,
    parsing sitemaps to extract relevant links based on last modification dates or target dates,
    and fetching news content from the extracted links.

    Args:
        base_url (str): The base URL of the website.
        date (str): The target date for news extraction in string format (e.g., 'YYYY-MM-DD').
        date_format (str): The format of the date string.

    Methods:
        get_sitemaps(): Retrieves sitemaps from the website's robots.txt file.
        parse_sitemap(sitemap): Parses a sitemap to extract links
                                based on last modification dates or target dates.
        extract_news(link): Extracts news content from a given link.
        main(delay): Executes the news extraction process.
    """

    def __init__(self, base_url: str, date: str, date_format: str):
        """
        Initialize a Finder instance.

        :param base_url: The base URL of the website.
        :param date: The target date for news extraction in string format.
        :param date_format: The format of the date string.
        """
        self.base_url = base_url.rstrip("/") if base_url.endswith("/") else base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        }
        self.date = date
        self.date_format = date_format
        self.date_formatted = datetime.strptime(date, date_format)
        self.year = str(self.date_formatted.year)
        self.month = "%02d" % self.date_formatted.month
        self.day = "%02d" % self.date_formatted.day

    def _get_robots(self) -> requests.Response:
        """
        Retrieve response from robots.txt

        :return: Response from robots.txt
        """
        robots = f"{self.base_url}/robots.txt"
        try:
            log.info("Fetching %s...", robots)
            response = requests.get(robots, headers=self.headers, timeout=60)
        except requests.exceptions.RequestException as err:
            log.exception("Unable to fetch url; err=%s", err)
        if response.status_code != 200:
            log.exception("robots.txt not available...")
        return response

    def get_sitemaps(self) -> str:
        """
        Retrieve sitemaps from the website's robots.txt file.
        If sitemap not in robots.txt, try base_url/sitemap.xml

        :return: A list of sitemap URLs.
        """
        response = self._get_robots()

        log.info("Retrieving sitemaps...")
        data = []
        lines = str(response.text).splitlines()
        for line in lines:
            if line.rstrip():
                if not line.startswith("#"):
                    split = line.split(":", maxsplit=1)
                    data.append([split[0].strip(), split[1].strip()])
        sitemaps = [
            directive[1] for directive in data if directive[0].lower() == "sitemap"
        ]
        if not sitemaps:
            log.warning("Sitemaps were not found via robots.txt")
            sitemaps = [f"{self.base_url}/sitemap.xml"]
            log.info("Trying to retrieve from %s...", sitemaps[0])
        return sitemaps

    def _parse_sitemap_with_lastmod(self, soup, links):
        """
        Parse a sitemap containing 'lastmod' tags to extract links based on last modification dates.

        :param soup: The BeautifulSoup object representing the sitemap XML.
        :return: A list of links from the sitemap that match the last modification date criteria.
        """
        # links = []
        lastmod_tags = soup.find_all("lastmod")
        for lastmod_tag in lastmod_tags:
            lastmod_date_str = lastmod_tag.text
            lastmod_date = datetime.strptime(lastmod_date_str, "%Y-%m-%dT%H:%M:%S%z")
            date_formatted_tz = self.date_formatted.replace(tzinfo=lastmod_date.tzinfo)
            if lastmod_date >= date_formatted_tz:
                loc_tag = lastmod_tag.find_previous("loc")
                if loc_tag:
                    link = loc_tag.text
                    date_pattern = rf"{self.year}.{{1}}?{self.month}.{{1}}?{self.day}"
                    if re.search(r".xml.*", link):
                        links += self.parse_sitemap(link)
                    elif re.search(date_pattern, link):
                        links.append(link)
        return links

    def _parse_sitemap_without_lastmod(self, soup, links):
        """
        Parse a sitemap without 'lastmod' tags to extract links based on the target date.

        :param soup: The BeautifulSoup object representing the sitemap XML.
        :return: A list of links from the sitemap that match the target date criteria.
        """
        # links = []
        loc_tags = soup.find_all("loc")
        for loc_tag in loc_tags:
            link = loc_tag.text
            date_pattern = rf"{self.year}.{{1}}?{self.month}.{{1}}?{self.day}"
            if re.search(date_pattern, link):
                if link.endswith(".xml"):
                    links += self.parse_sitemap(link)
                else:
                    links.append(link)
        return links

    def parse_sitemap(self, sitemap):
        """
        Parse a sitemap to extract links based on either 'lastmod' tags or the target date.

        :param sitemap: The URL of the sitemap to parse.
        :return: A list of links from the sitemap that match the date criteria.
        """
        log.info("Parsing sitemap: %s...", sitemap)
        response = requests.get(sitemap, headers=self.headers, timeout=60)
        soup = BeautifulSoup(response.content, "xml")
        links = []

        if soup.find("lastmod"):
            log.info("Last modified date found, parsing...")
            links = self._parse_sitemap_with_lastmod(soup, links)
        else:
            log.info("Last modified date not found, parsing via date...")
            links = self._parse_sitemap_without_lastmod(soup, links)

        unique_links = list(set(links))
        log.info("Found links: %s", len(unique_links))
        return unique_links

    def extract_news(self, link: str) -> str:
        """
        Extract news content from a given link using trafilatura.

        :param link: The URL of the news article.
        :return: Extracted news content as a string.
        """
        log.info("Fetching news from %s", link)
        response = trafilatura.fetch_url(link)
        if response:
            news = trafilatura.extract(response, favor_precision=True)
            log.info("News fetched!")
        if not response:
            log.exception('Failed to retrieve text from URL: "%s"', link)
        return news

    def main(self, delay: int = 10) -> tuple:
        """
        Main method for executing the news extraction process.

        :param delay: Delay in seconds between fetching news items.
        :return: A tuple containing extracted links and corresponding news items.
        """
        sitemaps = self.get_sitemaps()
        fetched_links = []
        news_items = []
        for sitemap in sitemaps:
            links = self.parse_sitemap(sitemap)
            if links:
                for link in links:
                    news = self.extract_news(link)
                    news_items.append(news)
                    fetched_links.append(link)
                    sleep(delay)
        if not fetched_links:
            log.exception("No news items to fetch!")
        return fetched_links, news_items
