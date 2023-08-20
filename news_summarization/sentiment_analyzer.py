"""
This module defines a class `SentimentAnalyzer` for running pretrained classifcation
pieplines for sentiment analysis and fetching their outputs.
"""
import logging

from transformers import pipeline

log = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A utility class for running pretrained classification pipelines for sentiment analysis
    and fetching their outputs.

    This class encapsulates the functionality of fetching a pretrained sentiment analysis model,
    running sentiment analysis on input text, and returning the analysis results.

    Args:
        model_name (str): The name of the pretrained sentiment analysis model.

    Methods:
        fetch_model(): Fetches the pretrained sentiment analysis model pipeline.
        run_model(classification_model, input_text): Runs sentiment analysis on input text.
    """
    def __init__(self, model_name: str):
        """
        Initialize a SentimentAnalyzer instance.

        :param model_name: The name of the pretrained sentiment analysis model.
        """
        self.model_name = model_name

    def fetch_model(self) -> pipeline:
        """
        Fetches the pretrained sentiment analysis model pipeline.

        :return: The fetched sentiment analysis model pipeline, or None in case of an error.
        """
        log.info('Fetching model=%s...', self.model_name)
        try:
            classification_model = pipeline(task="sentiment-analysis", model=self.model_name)
            log.info('Fetch succeded!')
            return classification_model
        except (OSError, ValueError) as err:
            log.error('Unknown exception when fetching model; err=%s', err)
            return None


    def run_model(self, classification_model: pipeline, input_text: str) -> dict:
        """
        Runs sentiment analysis on input text using the provided sentiment analysis model.

        :param classification_model: The pretrained sentiment analysis model pipeline.
        :param input_text: The text to perform sentiment analysis on.
        :return: The sentiment analysis results as a dictionary, or None in case of an error.
        """
        log.info('Running sentiment analysis...')
        try:
            output: dict = classification_model(input_text)[0]
            log.info('Sentiment analysis complete!')
            return output
        except (OSError, ValueError) as err:
            log.error('Unknown exception when running sentiment analysis; err=%s', err)
            return None
