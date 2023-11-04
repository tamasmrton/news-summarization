"""
This script fetches news articles from a specified news outlet's base URL,
summarizes the articles using NLP models, and saves the summarized data to Amazon S3.

It utilizes the Click library for command-line options and configuration, as well as
several external libraries for various tasks such as news extraction, summarization,
sentiment analysis, and data storage.
"""
import os
import sys
import logging
from urllib.parse import urlparse

import click
import tldextract
import pandas as pd
import awswrangler as wr
from dotenv import load_dotenv
from torch.multiprocessing import Pool
from torch.cuda import is_available

from news_summarization.news_extractor import NewsExtractor
from news_summarization.model_pipeline import ModelPipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(name)s: %(levelname)-4s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Global parameters
MAX_OUTPUT_LENGTH = 256
MIN_OUTPUT_LENGTH = 96
NUM_POOL_PROCESSES = 4
DELAY = 5
DEVICE = "cuda" if is_available() else "cpu"


class Workload:  # pylint: disable=too-many-instance-attributes
    """
    A Workload class representing the process of fetching news articles, summarizing them,
    and saving the summarized data to Amazon S3.

    This class encapsulates the workflow required to retrieve news articles from a specified
    news outlet's base URL, perform summarization using various NLP models, and analyze the
    sentiment of the summarized content. The summarized data is then structured and saved to
    Amazon S3 for further analysis or consumption.

    Args:
        base_url (str): The base URL of the news outlet.
        date (str): The date of news publication.

    Attributes:
        base_url (str): The base URL of the news outlet.
        date (str): The date of news publication.
    """

    def __init__(self, base_url: str, date: str, device: str):
        """
        Initialize the Workload instance.

        :param base_url: The base URL of the news outlet.
        :param date: The date of news publication.
        """
        self.base_url = base_url
        self.date = date
        self.device = device
        self.summarizer_model_name = None
        self.sentiment_model_name = None
        self.summarizer = None
        self.summarizer_model = None
        self.sentiment_analyzer = None
        self.sentiment_model = None

    def _initialize_models(self, summarizer_model_name: str, sentiment_model_name: str):
        """
        Initialize NLP models.

        :param summarizer_model_name: Summarizer model name.
        :param sentiment_model_name: Sentiment analyzer model name.
        """
        self.summarizer_model_name = summarizer_model_name
        self.sentiment_model_name = sentiment_model_name
        self.summarizer = ModelPipeline(summarizer_model_name, "summarization")
        self.summarizer_model = self.summarizer.fetch_model(device=self.device)
        self.sentiment_analyzer = ModelPipeline(
            sentiment_model_name, "sentiment-analysis"
        )
        self.sentiment_model = self.sentiment_analyzer.fetch_model()

    def _summarize_news(self, news: str) -> str:
        """
        Checks the length of the news item. If the token count reaches model limit,
        it splits the news into pieces and runs summarizer model on it.
        Lastly, it joins the result in a summary string output.

        :param news: News text to be summarized
        :return: Summarized text
        """
        news_split = self.summarizer.split_input_text(
            model=self.summarizer_model, input_text=news
        )
        summaries = []
        for news_item in news_split:
            summary_output = self.summarizer.run_model(
                model=self.summarizer_model,
                input_text=news_item,
                min_length=MIN_OUTPUT_LENGTH,
                max_length=MAX_OUTPUT_LENGTH,
            )

            if summary_output:
                summaries.append(summary_output["summary_text"])

        if not summaries:
            raise ValueError("No summary output created")

        news_summary = ("\n").join(summaries) if len(summaries) > 1 else summaries[0]
        return news_summary

    def _get_summarized_item(self, item: tuple) -> dict:
        """
        Run news text through summarizer and sentiment analyzer models.

        :param item: Tuple containing the news text and the link.
        :return: Dictionary containing article, the summary, sentiment score and model names.
        """
        link, news = item
        parsed_link = urlparse(link)
        summarized_item = {
            "source": parsed_link.netloc,
            "link": link,
            "article_text": news,
            "summary": None,
            "sentiment_label": None,
            "sentiment_score": None,
            "sentiment_model": None,
            "summarization_model": None,
        }
        try:
            log.info('Working on "%s"...', parsed_link.path)
            news_summary = self._summarize_news(news=news)

            sentiment_dict = self.sentiment_analyzer.run_model(
                self.sentiment_model, news_summary
            )
            summarized_item.update(
                {
                    "summary": news_summary,
                    "sentiment_label": sentiment_dict["label"],
                    "sentiment_score": sentiment_dict["score"],
                    "sentiment_model": self.sentiment_model_name,
                    "summarization_model": self.summarizer_model_name,
                }
            )
            return summarized_item
        except (IndexError, ValueError) as err:
            log.warning("Error in running the models; err=%s", err)
            return summarized_item
        except RuntimeError as err:
            if "CUDA" not in str(err):
                log.error("RuntimeError while running models; err=%s", err)
            log.warning("Error with CUDA process for running models; err=%s", err)
            return summarized_item

    def fetch_news_and_summarize(
        self,  # pylint: disable=too-many-locals
        date_format: str,
        summarizer_model_name: str,
        sentiment_model_name: str,
    ) -> list[dict]:
        """
        Fetch news, summarize, and analyze sentiment.

        :param date_format: Date format string.
        :param summarizer_model_name: Summarizer model name.
        :param sentiment_model_name: Sentiment analyzer model name.
        :return: List of summarized news items.
        """
        self._initialize_models(summarizer_model_name, sentiment_model_name)

        news_extractor = NewsExtractor(self.base_url, self.date, date_format)
        links, news_items = news_extractor.main(DELAY)
        items = zip(links, news_items)

        with Pool(processes=NUM_POOL_PROCESSES) as pool:
            results = pool.map(self._get_summarized_item, items)

        summarized_data = list(results)
        return summarized_data

    def save_to_s3(self, summarized_data: list[dict]):
        """
        Save summarized data to S3.

        :param summarized_data: List of summarized news items.
        """
        summarized_news_df = pd.DataFrame.from_dict(summarized_data)
        tld = tldextract.extract(self.base_url)
        s3_path = f's3://{os.getenv("AWS_BUCKET")}/{self.date}/{tld.suffix}/{tld.domain}.parquet'
        if len(summarized_data) > 0:
            wr.s3.to_parquet(summarized_news_df, s3_path)
            log.info("Data saved to path: %s", s3_path)


@click.command()
@click.option(
    "--base-url",
    "base_url",
    default=None,
    type=str,
    help="The base url of the news outlet.",
)
@click.option("--date", default=None, type=str, help="The date of news publication.")
@click.option(
    "--date-format",
    "date_format",
    default="%Y-%m-%d",
    type=str,
    help="Date format, i.e. `%Y-%m-%d`.",
)
@click.option(
    "--summarizer-model",
    "summarizer_model_name",
    default=None,
    type=str,
    help="Summarizer model name.",
)
@click.option(
    "--sentiment-model",
    "sentiment_model_name",
    default=None,
    type=str,
    help="Sentiment analyzer model name.",
)
@click.option(
    "--device",
    "device",
    default=DEVICE,
    type=click.Choice(["cuda", "cpu"]),
    help="Processing unit; cuda or cpu",
)
def main(  # pylint: disable=too-many-arguments
    base_url: str,
    date: str,
    date_format: str,
    summarizer_model_name: str,
    sentiment_model_name: str,
    device: str,
):
    """
    Main entry point for the news summarization and saving process.

    :param base_url: The base URL of the news outlet.
    :param date: The date of news publication.
    :param date_format: Date format string.
    :param summarizer_model_name: Summarizer model name.
    :param sentiment_model_name: Sentiment analyzer model name.
    """
    workload = Workload(base_url, date, device)
    summarized_data = workload.fetch_news_and_summarize(
        date_format, summarizer_model_name, sentiment_model_name
    )
    workload.save_to_s3(summarized_data)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
