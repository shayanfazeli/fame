from typing import List, Union
from fame.text_processing.language import is_from_these_languages

from fame.text_processing.function_bank import TextProcessingMethodBank


class TextProcessor:
    def __init__(
            self,
            methods: List[str] = TextProcessingMethodBank.methods,
            allowed_languages: List[str] = {'English'}
    ):
        self.methods = methods
        self.allowed_languages = allowed_languages

    def process_text_with_methods(self, text: str) -> str:
        if not is_from_these_languages(text, self.allowed_languages):
            return None
        for method in self.methods:
            text = getattr(TextProcessingMethodBank, method)(text=text)
        return text

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return self.process_text_with_methods(text)
        elif isinstance(text, List):
            output = [self.process_text_with_methods(e) for e in text]
            return [e for e in output if e is not None]
        else:

            raise Exception(f"we cannot process type: {type(text)}")
