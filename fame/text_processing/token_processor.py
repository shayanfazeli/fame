from typing import List, Union
import nltk
import re

from fame.text_processing.function_bank import TokenProcessingMethodBank


class TokenProcessor:
    def __init__(
            self,
            methods: List[str] = TokenProcessingMethodBank.methods
    ):
        self.methods = methods

    def segment_to_word(self, text: str) -> List[str]:
        """
        Parameters
        ----------
        text: `str`, required
            The text

        Returns
        ----------
        List of tokens will be returned
        """
        if text is None:
            return None
        else:
            word_list = nltk.word_tokenize(text)
        return word_list

    def process_tokens(self, word_list: List[str]) -> List[str]:
        """
        Parameters
        ----------
        word_list: `List[str]`, required
            The list of words

        Returns
        ----------
        The list of processed tokens
        """
        for method in self.methods:
            word_list = getattr(TokenProcessingMethodBank, method)(word_list=word_list)

        return word_list

    def __call__(self, input_data: Union[str, List[str]]) -> List[str]:
        """
        Parameters
        ----------
        input_data: `Union[str, List[str]]`, required
            The text to tokenize, and then process the tokens, or the tokens to be directly processed.

        Returns
        ----------
        The list of processed tokens
        """
        if input_data is None:
            return None

        if isinstance(input_data, List):
            return self.process_tokens(word_list=input_data)
        elif isinstance(input_data, str):
            return self.process_tokens(word_list=self.segment_to_word(text=input_data))
        else:
            raise Exception(f"we cannot process type: {type(input_data)}")
