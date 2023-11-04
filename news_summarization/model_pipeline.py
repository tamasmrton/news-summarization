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

    @staticmethod
    def _find_middle(splitted_text: list) -> int:
        """
        Find the index of the middle element in a list.

        :param splitted_text: A list of elements.
        :type splitted_text: list
        :return: The index of the middle element in the list. If the list has an even number of elements,
                 this will be the index of the first element in the right half of the list.
        """
        length = len(splitted_text)
        middle_index = length // 2
        return middle_index

    @staticmethod
    def _count_tokens(model: pipeline, input_text: str) -> int:
        """
        Counts number of tokens in input text.

        :param model: The pretrained model pipeline.
        :param input_text: The text to count the tokens on.
        :return: The number of tokens in the text.
        """
        log.info("Count number of tokens in text...")
        tokens = len(model.tokenizer(input_text).input_ids)
        return tokens

    def fetch_model(self, device: str = "cpu") -> pipeline:
        """
        Fetches the pretrained model pipeline.

        :param device: Use CPU or GPU for processing.
        :return: The fetched model pipeline, or None in case of an error.
        """
        log.info("Fetching model=%s...", self.model_name)
        log.info("Using %s...", device)
        try:
            model = pipeline(task=self.task, model=self.model_name, device=device)
            log.info("Fetch succeded!")
            return model
        except (OSError, ValueError) as err:
            log.error("Unknown exception when fetching model; err=%s", err)
            return None

    def split_input_text(self, model: pipeline, input_text: str) -> list:
        """
        Splits input text into two parts based on newline character.
        Only does so when model token limit is reached.

        :param model: The pretrained model pipeline.
        :param input_text: The text to count the tokens on.
        :return: List of input texts.
        """
        tokens = self._count_tokens(model, input_text)
        max_tokens = model.tokenizer.model_max_length
        if tokens > max_tokens:
            log.info("Maximum allowed tokens reached; [%s > %s]", tokens, max_tokens)
            log.info("Splitting text to smaller chunks...")
            splitted_text = input_text.split("\n")
            middle_index = self._find_middle(splitted_text)
            first_part = splitted_text[:middle_index]
            second_part = splitted_text[middle_index:]
            return ["\n".join(first_part), "\n".join(second_part)]
        log.info("Number of tokens in input is less than model capacity, continuing...")
        return [input_text]

    def run_model(self, model: pipeline, input_text: str, **kwargs) -> dict:
        """
        Runs model on input text using the provided model name.

        :param model: The pretrained model pipeline.
        :param input_text: The text to perform text analysis on.
        :return: The text analysis results as a dictionary, or None in case of an error.
        """
        log.info("Running %s...", self.task.capitalize())
        try:
            output: dict = model(input_text, **kwargs)[0]
            log.info("%s complete!", self.task.capitalize())
            return output
        except (OSError, ValueError) as err:
            log.error("Unknown exception when running %s; err=%s", self.task, err)
            return None
