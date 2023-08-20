"""
This module defines a class `Summarizer` for running pretrained tokenizer
models and fetching their outputs.
"""
import logging

from transformers import PreTrainedModel, PreTrainedTokenizer

log = logging.getLogger(__name__)

class Summarizer:
    """
    A utility class for summarizing text using pretrained language models.

    This class encapsulates the functionality of fetching a pretrained tokenizer and generator,
    as well as running a summarization model on input text to produce concise summaries.

    Attributes:
        MAX_INPUT_LENGTH (int): The maximum length of input text.
        MAX_OUTPUT_LENGTH (int): The maximum length of the generated summary.
        MIN_OUTPUT_LENGTH (int): The minimum length of the generated summary.

    Args:
        model_generator (PreTrainedModel): The generator model for summarization.
        model_name (str): The name of the pretrained model.
        tokenizer (PreTrainedTokenizer): The tokenizer used to preprocess text.

    Methods:
        fetch_model(): Fetches the pretrained tokenizer and generator for the specified model.
        run_model(pretrained_tokenizer, pretrained_model, input_text):
            Runs the summarization model on input text.
    """
    MAX_INPUT_LENGTH = 1024
    MAX_OUTPUT_LENGTH = 256
    MIN_OUTPUT_LENGTH = 128

    def __init__(self,
                 model_generator: PreTrainedModel,
                 model_name: str,
                 tokenizer: PreTrainedTokenizer):
        """
        Initialize a Summarizer instance.

        :param model_generator: The generator model for summarization.
        :param model_name: The name of the pretrained model.
        :param tokenizer: The tokenizer used to preprocess text.
        """
        self.model_generator = model_generator
        self.model_name = model_name
        self.tokenizer = tokenizer

    def fetch_model(self):
        """
        Fetch the pretrained tokenizer and generator for the specified model.

        :return: A tuple containing the pretrained tokenizer and generator models.
        """
        log.info('Fetching tokenizer=%s and generator=%s for model=%s...',
                  getattr(self.tokenizer, '__name__', 'Unknown'),
                  getattr(self.model_generator, '__name__', 'Unknown'),
                  self.model_name)
        try:
            pretrained_tokenizer = self.tokenizer.from_pretrained(self.model_name)
            pretrained_model = self.model_generator.from_pretrained(self.model_name)
            log.info('Fetch succeded!')
            return pretrained_tokenizer, pretrained_model
        except (OSError, ValueError) as err:
            log.error('Unknown exception when fetching model; err=%s', err)
            return None

    def run_model(self,
                  pretrained_tokenizer: PreTrainedTokenizer,
                  pretrained_model: PreTrainedModel,
                  input_text: str):
        """
        Run the summarization model on the input text.

        :param pretrained_tokenizer: The pretrained tokenizer.
        :param pretrained_model: The pretrained generator model.
        :param input_text: The text to be summarized.
        :return: The summarized text.
        """
        log.info('Running summarization...')
        try:
            tokenized_text = pretrained_tokenizer(input_text,
                                                  truncation=True,
                                                  max_length=self.MAX_INPUT_LENGTH,
                                                  return_tensors="pt")
            summarization = pretrained_model.generate(**tokenized_text,
                                                      max_length=self.MAX_OUTPUT_LENGTH,
                                                      min_length=self.MIN_OUTPUT_LENGTH)
            summed_text = pretrained_tokenizer.batch_decode(summarization, skip_special_tokens=True)
            log.info('Summarization complete!')
            return summed_text[0]
        except (OSError, ValueError) as err:
            log.error('Unknown exception when running summarization; err=%s', err)
            return None
