from typing import List, Union
from fame.text_processing.language import is_from_these_languages

from fame.text_processing.function_bank import TextProcessingMethodBank


class TextProcessor:
    """
    Parameters
    ----------
    methods: `List[str]`, optional(default=`TextProcessingMethodBank.methods`)
        The list of methods which will be applied by the order in this list to the text.
        The :class:`TextProcessingMethodBank` contains the methods to select from.

    allowed_languages: `List[str]`, optional(default=`{'English'}`)
        The allowed languages
    """
    def __init__(
            self,
            methods: List[str] = TextProcessingMethodBank.methods,
            allowed_languages: List[str] = {'English'}
    ):
        """
        constructor
        """
        self.methods = methods
        self.allowed_languages = allowed_languages

    def process_text_with_methods(self, text: str) -> str:
        """
        Parameters
        ----------
        text: `str`, required
            The text to be processed

        Returns
        ----------
        The processed text by the methods provided to this object.
        """
        if not is_from_these_languages(text, self.allowed_languages):
            return None
        for method in self.methods:
            text = getattr(TextProcessingMethodBank, method)(text=text)
        return text

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Parameters
        ----------
        text: `Union[str, List[str]]`, required
            The single or list of texts to be processed.

        Returns
        ----------
        If a string is provided, it will be processed and if a list of texts,
            the list of processed texts will be returned.
        """
        if isinstance(text, str):
            return self.process_text_with_methods(text)
        elif isinstance(text, List):
            output = [self.process_text_with_methods(e) for e in text]
            return [e for e in output if e is not None]
        else:
            raise Exception(f"we cannot process type: {type(text)}")
