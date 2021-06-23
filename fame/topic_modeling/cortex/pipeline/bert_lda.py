from typing import List, Tuple, Union
import random

from tqdm import tqdm
import gensim

import numpy
import torch
import torch.nn
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from fame.text_processing.text_processor import TextProcessor
from fame.text_processing.token_processor import TokenProcessor
from fame.topic_modeling.cortex.model.autoencoder import MLPAutoEncoder


class TransformerLDATopicModelingPipeline:
    def __init__(
            self,
            logdir: str,
            autoencoder: MLPAutoEncoder = None,
            use_transformer: bool = True,
            use_lda: bool = True,
            number_of_topics_for_lda: int = 10,
            use_tfidf: bool = False,
            transformer_modelname: str = 'paraphrase-mpnet-base-v2',
            device: torch.device = torch.device('cpu'),
            text_processor: TextProcessor = TextProcessor(),
            token_processor_light: TokenProcessor = TokenProcessor(
                methods=[
                    'keep_alphabetics_only',
                    # 'keep_nouns_only',
                    'spell_check_and_typo_fix',
                    # 'stem_words',
                    # 'remove_stopwords'
                ]
            ),
            token_processor_heavy: TokenProcessor = TokenProcessor(
                methods=[
                    'keep_alphabetics_only',
                    # 'keep_nouns_only',
                    'spell_check_and_typo_fix',
                    'stem_words',
                    'remove_stopwords'
                ]
            ),
            pca: PCA = None,
            representation_clustering: Union[KMeans, DBSCAN] = KMeans(n_clusters=5, random_state=1010)
    ):
        self.logdir = logdir

        self.device = device

        self.use_tfidf = use_tfidf
        self.use_transformer = use_transformer
        self.transformer = SentenceTransformer(transformer_modelname).to(device) if use_transformer else None
        self.autoencoder = autoencoder.to(device)
        self.autoencoder_is_trained = False

        self.tfidf = TfidfVectorizer() if use_tfidf else None
        self.tfidf_is_trained = False

        self.text_processor = text_processor
        self.token_processor_light = token_processor_light
        self.token_processor_heavy = token_processor_heavy

        self.pca_is_trained = False
        self.pca = pca

        self.use_lda = use_lda
        self.lda_model = None
        self.number_of_topics_for_lda = number_of_topics_for_lda

        self.representation_clustering_is_trained = False
        self.representation_clustering = representation_clustering

    def preprocess_and_get_text_and_tokens(
            self,
            text_list: List[str],
            random_sample_count: int = None,
            replace: bool = False
    ):
        if random_sample_count is not None:
            sampling_indices = numpy.random.choice(len(text_list), random_sample_count, replace=replace)
        else:
            sampling_indices = numpy.arange(len(text_list))

        preprocessed_text_list = []
        preprocessed_tokens_list = []
        indices = []

        for i, sampling_index in enumerate(tqdm(sampling_indices)):
            text = text_list[sampling_index]
            try:
                preprocessed_text_list.append(
                    ' '.join(self.token_processor_light(self.text_processor(text)))
                )
                preprocessed_tokens_list.append(self.token_processor_heavy(self.text_processor(text)))
                indices.append(sampling_index)
            except Exception as e:
                continue

        return preprocessed_text_list, preprocessed_tokens_list, indices

    def prepare_lda_model(
            self,
            tokens_list: List[List[str]],
            lda_worker_count: int = 1
    ):
        if self.lda_model is not None:
            return self.lda_model

        self.vocabulary = gensim.corpora.Dictionary(tokens_list)
        self.corpus = [self.vocabulary.doc2bow(tokens) for tokens in tokens_list]

        self.lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=self.corpus,
            num_topics=self.number_of_topics_for_lda,
            id2word=self.vocabulary,
            passes=20,
            chunksize=10000,
            workers=lda_worker_count,
            per_word_topics=True
        )

    def prepare_tfidf_model(self, text_list: List[str]):
        self.tfidf.fit(text_list)
        self.tfidf_is_trained = True

    def get_lda_representations(
            self,
            tokens_list: List[List[str]]
    ):
        assert self.lda_model is not None, "prepare the lda model first"
        corpus = [self.vocabulary.doc2bow(text) for text in tokens_list]
        lda_representations = numpy.zeros((len(corpus), self.number_of_topics_for_lda))
        for i in range(len(corpus)):
            # get the distribution for the i-th document in corpus
            for topic, prob in self.lda_model.get_document_topics(corpus[i]):
                lda_representations[i, topic] = prob
        return lda_representations

    def get_stacked_representations(self, text_list, tokens_list, reduce_dimensions: bool = False):
        reps = []
        if self.use_lda:
            assert self.lda_model is not None, "train the LDA first"
            lda_reps = self.get_lda_representations(tokens_list=tokens_list)
            reps.append(lda_reps)

        if self.use_tfidf:
            assert self.tfidf_is_trained, "train the tfidf model first"
            reps.append(self.tfidf.transform(text_list))

        if self.use_transformer:
            reps.append(self.transformer.encode(text_list))

        if len(reps) > 1:
            reps = numpy.concatenate(reps, axis=1)

        if reduce_dimensions:
            assert self.pca is not None and self.pca_is_trained, "a trained PCA model is needed for dimensionality reduction"
            reps = self.pca.transform(reps)

        return reps

    def prepare_autoencoder(self, text_list, tokens_list, number_of_epochs: int = 200, batch_size: int = 128, shuffle: bool =True):
        if self.autoencoder_is_trained:
            return

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-6)
        reps = self.get_stacked_representations(text_list=text_list, tokens_list=tokens_list)

        if shuffle:
            numpy.random.shuffle(reps)

        epoch_losses = []
        epoch_index_range = tqdm(range(number_of_epochs))
        for epoch_index in epoch_index_range:
            batch_losses = []
            for i in range(len(reps) // batch_size + 1):
                optimizer.zero_grad()
                start_index = batch_size * i
                end_index = batch_size * (i + 1)
                if end_index > reps.shape[0]:
                    end_index = reps.shape[0]
                x = torch.from_numpy(reps[numpy.arange(start_index, end_index), :]).to(self.device).float()
                loss = self.autoencoder(x, return_loss=True)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            epoch_losses.append(numpy.mean(batch_losses))
            epoch_index_range.set_description(f"Epoch: {epoch_index} / Loss: {epoch_losses[-1]}")
            epoch_index_range.refresh()

        return epoch_losses

    def get_representations(self, text_list: List[str], batch_size=128) -> numpy.ndarray:
        preprocessed_text_list, preprocessed_tokens_list, indices = self.preprocess_and_get_text_and_tokens(
            text_list)

        reps = self.get_stacked_representations(text_list = preprocessed_text_list, tokens_list=preprocessed_tokens_list)

        embeddings = []
        for i in range(reps.shape[0] // batch_size + 1):
            start_index = batch_size * i
            end_index = batch_size * (i + 1)
            x = torch.from_numpy(reps[numpy.arange(start_index, end_index), :]).to(self.device).float()
            embeddings.append(self.autoencoder(x, return_embeddings=True))
        embeddings = torch.cat(embeddings, dim=0).data.cpu().numpy()

        return embeddings

    def train_clustering(self, reps) -> None:
        self.representation_clustering.fit(reps)
        self.representation_clustering_is_trained = True

    def label_representation_cluster(self, reps):
        assert self.representation_clustering_is_trained, "clustering model is not trained yet"
        return self.representation_clustering.predict(reps)

    def train_pca(self, reps) -> None:
        self.pca.fit(reps)
        self.pca_is_trained = True
