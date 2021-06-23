from typing import List
import re
import nltk
import pkg_resources
from nltk.stem.porter import PorterStemmer
# stemming if doing word-wise
p_stemmer = PorterStemmer()

from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

from stop_words import get_stop_words
stop_words = (list(
    set(get_stop_words('en'))
))


class TextProcessingMethodBank:
    methods = [
        'remove_url',
        'convert_to_lowercase',
        'uppercase_based_missing_delimiter_fix',
        'gtlt_normalize',
        'substitute_more_than_two_letter_repetition_with_one',
        'non_character_repetition_elimination',
        'use_star_as_delimiter',
        'remove_parantheses_and_their_contents',
        'remove_questionexlamation_in_brackets',
        # 'eliminate_phrase_repetition',
        'strip'
    ]

    @staticmethod
    def remove_url(text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+.com", "", text)

        return text

    @staticmethod
    def uppercase_based_missing_delimiter_fix(text: str) -> str:
        return re.sub(r'([a-z])([A-Z])', r'\1. \2', text)

    @staticmethod
    def convert_to_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def gtlt_normalize(text: str) -> str:
        return re.sub(r'&gt|&lt', ' ', text)

    @staticmethod
    def substitute_more_than_two_letter_repetition_with_one(text: str) -> str:
        return re.sub(r'([a-z])\1{2,}', r'\1', text)

    @staticmethod
    def non_character_repetition_elimination(text: str) -> str:
        return re.sub(r'([\W+])\1{1,}', r'\1', text)

    @staticmethod
    def use_star_as_delimiter(text: str) -> str:
        return re.sub(r'\*|\W\*|\*\W', '. ', text)

    @staticmethod
    def remove_parantheses_and_their_contents(text: str) -> str:
        return re.sub(r'\(.*?\)', '. ', text)

    @staticmethod
    def remove_questionexlamation_in_brackets(text: str) -> str:
        text = re.sub(r'\W+?\.', '.', text)
        text = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', text)
        return text

    @staticmethod
    def eliminate_phrase_repetition(text: str) -> str:
        return re.sub(r'(.{2,}?)\1{1,}', r'\1', text)

    @staticmethod
    def strip(text: str) -> str:
        return text.strip()


class TokenProcessingMethodBank:
    methods = [
        'keep_alphabetics_only',
        'keep_nouns_only',
        'spell_check_and_typo_fix',
        'stem_words',
        'remove_stopwords'
    ]
    @staticmethod
    def keep_alphabetics_only(word_list: List[str]) -> List[str]:
        return [word for word in word_list if word.isalpha()]

    @staticmethod
    def keep_nouns_only(word_list: List[str]) -> List[str]:
        return [word for (word, pos) in nltk.pos_tag(word_list) if pos[:2] == 'NN']

    @staticmethod
    def spell_check_and_typo_fix(word_list: List[str]) -> List[str]:
        outputs = []
        for word in word_list:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                outputs.append(suggestions[0].term)
            else:
                pass

        return outputs

    @staticmethod
    def stem_words(word_list: List[str]) -> List[str]:
        return [p_stemmer.stem(word) for word in word_list]

    @staticmethod
    def remove_stopwords(word_list: List[str]) -> List[str]:
        return [word for word in word_list if word not in stop_words]
