from typing import List

import language_detector


def is_english(text: str) -> bool:
    return language_detector.detect_language(text) in {'Engish'}


def is_from_these_languages(text: str, language_list: List[str]) -> bool:
    return language_detector.detect_language(text) in language_list


def translate(text: str, to: str) -> str:
    raise NotImplementedError
