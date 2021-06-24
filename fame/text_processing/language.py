from typing import List

import language_detector


def is_english(text: str) -> bool:
    """
    Parameters
    ----------
    text: `str`, required
        Text piece

    Returns
    ----------
    Returns True if english
    """
    return language_detector.detect_language(text) in {'Engish'}


def is_from_these_languages(text: str, language_list: List[str]) -> bool:
    """
    Parameters
    ----------
    text: `str`, required
        Text piece

    language_list: `List[str]`, required
        List of languages

    Returns
    ----------
    Returns true if the text is from the given languages (the given languages must be compliant with the
    `language_detector` package, which is used in this method.
    """
    return language_detector.detect_language(text) in language_list


def translate(text: str, to: str) -> str:
    """
    Parameters
    ----------
    text: `str`, required
        Input text
    to: `str`, required
        The target language

    Returns
    ----------
    Translated text
    """
    raise NotImplementedError
