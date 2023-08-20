"""
This module defines a class `ModelPipeline` for running pretrained
pieplines for text analysis and fetching their outputs.
"""
import logging

from transformers import pipeline

log = logging.getLogger(__name__)

class ModelPipeline:
    """
    A utility class for running pretrained pipelines and fetching their outputs.

    This class encapsulates the functionality of fetching a pretrained model,
    running analysis on input text, and returning the text analysis results.

    Args:
        model_name (str): The name of the pretrained model.

    Methods:
        fetch_model(): Fetches the pretrained model pipeline.
        run_model(classification_model, input_text): Runs analysis on input text.
    """
    def __init__(self, model_name: str, task: str):
        """
        Initialize a ModelPipeline instance.

        :param model_name: The name of the pretrained text analysis model.
        """
        self.model_name = model_name
        self.task = task

    def fetch_model(self) -> pipeline:
        """
        Fetches the pretrained model pipeline.

        :return: The fetched model pipeline, or None in case of an error.
        """
        log.info('Fetching model=%s...', self.model_name)
        try:
            model = pipeline(task=self.task, model=self.model_name)
            log.info('Fetch succeded!')
            return model
        except (OSError, ValueError) as err:
            log.error('Unknown exception when fetching model; err=%s', err)
            return None


    def run_model(self, model: pipeline, input_text: str, **kwargs) -> dict:
        """
        Runs model on input text using the provided model name.

        :param model: The pretrained model pipeline.
        :param input_text: The text to perform text analysis on.
        :return: The text analysis results as a dictionary, or None in case of an error.
        """
        log.info('Running %s...', self.task.capitalize())
        try:
            output: dict = model(input_text, **kwargs)[0]
            log.info('%s complete!', self.task.capitalize())
            return output
        except (OSError, ValueError) as err:
            log.error('Unknown exception when running %s; err=%s', self.task, err)
            return None
