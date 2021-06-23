import unittest
from fame.text_processing.token_processor import TokenProcessor
from fame.text_processing.text_processor import TextProcessor


class TextProcessingTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_text_processing(self):
        text_processor = TextProcessor()
        sample_text = """
                my world has colllappsed and i understand that this world is www.google.com http://google.com and $TESLA and #$APPLE are  fun..    #$!? you tell me ?
                """
        self.assertEqual(text_processor(sample_text), 'my world has colappsed and i understand that this world is and $tesla and #$apple are fun. #$!? you tell me ?')

    def test_token_processing(self):
        sample_text = """
                        my world has colllappsed and i understand that this world is www.google.com http://google.com and $TESLA and #$APPLE are  fun..    #$!? you tell me ?
                        """
        text_processor = TextProcessor()
        token_processor = TokenProcessor(
            methods=[
                'keep_alphabetics_only',
                # 'keep_nouns_only',
                'spell_check_and_typo_fix',
                # 'stem_words',
                # 'remove_stopwords'
            ]
        )
        self.assertEqual(
            ' '.join(token_processor(text_processor(sample_text))),
            'my world has collapsed and i understand that this world is and tesla and apple are fun you tell me')

    def test_stemming_level_token_processing(self):
        sample_text = """
                                my world has colllappsed and i understand that this world is www.google.com http://google.com and $TESLA and #$APPLE are  fun..    #$!? you tell me ?
                                """
        text_processor = TextProcessor()
        token_processor = TokenProcessor(
            methods=[
                'keep_alphabetics_only',
                # 'keep_nouns_only',
                'spell_check_and_typo_fix',
                'stem_words',
                'remove_stopwords'
            ]
        )
        self.assertEqual(
            ' '.join(token_processor(text_processor(sample_text))),
            'world ha collaps understand thi world tesla appl fun tell')