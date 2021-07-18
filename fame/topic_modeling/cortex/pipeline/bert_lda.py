from typing import List, Tuple, Union, Dict
import random
from tqdm import tqdm

import numpy
import torch
import torch.nn
from sentence_transformers import SentenceTransformer

import gensim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from fame.text_processing.text_processor import TextProcessor
from fame.text_processing.token_processor import TokenProcessor
from fame.topic_modeling.cortex.model.autoencoder import MLPAutoEncoder


class TransformerLDATopicModelingPipeline:
    """
    The :class:`TransformerLDATopicModelingPipeline` class provides an easy-to-use wrapper for
    performing neural topic modeling with the state of the art transformers, clustering techniques, and
    topic modeling based on latent dirichlet analysis.

    Please refer to [this document](#todo: fill) for an example usage.

    Parameters
    ----------
    autoencoder: `MLPAutoEncoder`, optional (default=None)
        If provided (with a value other than `None`), the so-called "stacked" representations will be
        passed through this VAE and the dense bottleneck representations of the learned auto-encoder
        will be the final representations.

    use_transformer: `bool`, optional (default=True)
        Boolean flag indicating whether or not we want "transformer" features to contribute to the "stacked" representations.

    use_lda: `bool`, optional (default=True)
        Boolean flag indicating whether or not we want "LDA" features to contribute to the "stacked"
        representations (the features indicate the probability layout of topics conditioned by the given document).

    number_of_topics_for_lda: `int`, optional (default=10)
        If provided, this will be the number of latent dirichlet allocation based topics

    use_tfidf: `bool`, optional (default=False)
        Boolean flag indicating whether or not we want "tf-idf" features to contribute to the "stacked" representations.

    transformer_modelname: `str`, optional(default='paraphrase-mpnet-base-v2')
        The pretrained sequence embedding model based on [this list](https://www.sbert.net/docs/pretrained_models.html).

    device: `torch.device`, optional (default=`torch.device('cpu')`)
        The instance which will be used for torch modules

    text_processor: `TextProcessor`, optional (default=`TextProcessor()`)
        The module for processing string, which combined with `token_processor_light` will be the main text
        preprocessing scheme.

    token_processor_light: `TokenProcessor`, optional (default=```
        TokenProcessor = TokenProcessor(
            methods=[
                'keep_alphabetics_only',
                # 'keep_nouns_only',
                'spell_check_and_typo_fix',
                # 'stem_words',
                # 'remove_stopwords'
            ]
        )
        ```)
        Token processor (light)

    token_processor_heavy: `TokenProcessor`, optional (default=```
        TokenProcessor = TokenProcessor(
            methods=[
                'keep_alphabetics_only',
                # 'keep_nouns_only',
                'spell_check_and_typo_fix',
                'stem_words',
                'remove_stopwords'
            ]
        )
        ```)
        Token processor (heavy) - Please note that a more thorough token preprocessing (including stemming etc.)
        is required for the LDA approach.

    pca: `PCA`, optional (default=None)
        The PCA module for dimensionality reduction. If provided, the "apply_pca_on_stacked_reps" can be used in
        the :meth:`get_stacked_representations` to apply it to the representations.

    representation_clustering: `Union[KMeans, DBSCAN]`, optional(default=`KMeans(n_clusters=5, random_state=1010)`)
        The representation clustering model
    """
    def __init__(
            self,
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
            apply_pca_on_stacked_reps: bool = False,
            representation_clustering: Union[KMeans, DBSCAN] = KMeans(n_clusters=5, random_state=1010)
    ):
        """
        constructor
        """
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
        self.apply_pca_on_stacked_reps = apply_pca_on_stacked_reps

        self.use_lda = use_lda
        self.lda_model = None
        self.number_of_topics_for_lda = number_of_topics_for_lda

        self.representation_clustering_is_trained = False
        self.representation_clustering = representation_clustering

    def preprocess_and_get_text_and_tokens(
            self,
            text_list: List[str],
            random_sample_count: int = None,
            replace: bool = False,
            verbose: bool = False
    ) -> Tuple[List[str], List[List[str]], List[int]]:
        """
        Parameters
        ----------
        text_list: `List[str]`, required
            The list of input texts

        random_sample_count: `int`, optional (default=None)
            If provided, it will be the number of text elements to be randomly sampled from this list.

        replace: `bool`, optional (default=False)
            It will be a parameter used when random sampling is chosen.

        Returns
        ----------
        It returns the following outputs:
        `preprocessed_text_list, preprocessed_tokens_list, indices`

        The first element:
        * the list of preprocessed versions of the original
        * the second element is the list of token lists, per each element
        * list of original ijndices
        """
        if random_sample_count is not None:
            sampling_indices = numpy.random.choice(len(text_list), random_sample_count, replace=replace)
        else:
            sampling_indices = numpy.arange(len(text_list))

        preprocessed_text_list = []
        preprocessed_tokens_list = []
        indices = []

        the_iterator = tqdm(sampling_indices) if verbose else sampling_indices
        for i, sampling_index in enumerate(the_iterator):
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
    ) -> None:
        """
        Parameters
        ----------
        tokens_list: `List[List[str]]`, required
            The list of token lists

        lda_worker_count: `int`, optional (default=1)
            The number of workers in the lda module.

        :return: 
        """
        if self.lda_model is not None:
            return self.lda_model

        self.vocabulary = gensim.corpora.Dictionary(tokens_list)
        self.corpus = [self.vocabulary.doc2bow(tokens) for tokens in tokens_list]

        self.lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=tqdm(self.corpus),
            num_topics=self.number_of_topics_for_lda,
            id2word=self.vocabulary,
            passes=20,
            chunksize=10000,
            workers=lda_worker_count,
            per_word_topics=True
        )

    def prepare_tfidf_model(self, text_list: List[str]) -> None:
        """
        Parameters
        ----------
        text_list: `List[str]`, required
            The list of texts (please note that it is caller's responsibility to do any preprocessing beforehand).
        """
        self.tfidf.fit(text_list)
        self.tfidf_is_trained = True

    def get_lda_representations(
            self,
            tokens_list: List[List[str]]
    ) -> numpy.ndarray:
        """
        Parameters
        ----------
        tokens_list: `List[List[str]]`, required
            The list of token lists

        Returns
        ----------
        The `numpy.ndarray` of dimensions: `len(token_list), number_of_topics_for_lda` corresponding to the
        topic probability layout per documents.
        """
        assert self.lda_model is not None, "prepare the lda model first"
        corpus = [self.vocabulary.doc2bow(text) for text in tokens_list]
        lda_representations = numpy.zeros((len(corpus), self.number_of_topics_for_lda))
        for i in range(len(corpus)):
            # get the distribution for the i-th document in corpus
            for topic, prob in self.lda_model.get_document_topics(corpus[i]):
                lda_representations[i, topic] = prob
        return lda_representations

    def get_stacked_representations(self, text_list: List[str], tokens_list: List[List[str]], batch_size: int = 128) -> numpy.ndarray:
        """
        Parameters
        ----------
        text_list: `List[str]`, required
            The list of texts (please note that it is caller's responsibility to do any preprocessing beforehand).

        tokens_list: `List[List[str]]`, required
            The list of token lists

        batch_size: `int`, optional (default=128)
            The batch-size for computationally heavier modules.

        Returns
        ----------
        Prior to auto-encoder (if any auto-encoder is provided), the tfidf, lda, and transformer based representations
        (if used, which is indicated by the corresponding boolean flags) will be computed, concatenated, and passed.
        PCA,if provided and if its use for stacked_representaitons is also indicated by setting the
        `apply_pca_on_stacked_reps`, it will also be applkiedf on the representations.
        """
        reps = []
        if self.use_lda:
            assert self.lda_model is not None, "train the LDA first"
            lda_reps = self.get_lda_representations(tokens_list=tokens_list)
            if lda_reps.shape[0] == 0:
                raise Exception("no token list")
            if lda_reps.ndim == 1:
                lda_reps = lda_reps.reshape(1, -1)
            reps.append(lda_reps)

        if self.use_tfidf:
            assert self.tfidf_is_trained, "train the tfidf model first"
            reps.append(self.tfidf.transform(text_list))

        if self.use_transformer:
            if len(text_list) <= batch_size:
                transformer_reps = self.transformer.encode(text_list)
                if transformer_reps.ndim == 1:
                    transformer_reps = transformer_reps.reshape(1, -1)
            else:
                transformer_reps = []
                for i in range(len(text_list) // batch_size + 1):
                    start_index = batch_size * i
                    if start_index == len(text_list):
                        break
                    end_index = batch_size * (i + 1)
                    if end_index > len(text_list):
                        end_index = len(text_list)
                    tmp = self.transformer.encode(text_list[start_index:end_index])
                    if tmp.ndim == 1:
                        tmp = tmp.reshape(1, -1)
                    transformer_reps.append(tmp)
                transformer_reps = numpy.concatenate(transformer_reps, axis=0)
            reps.append(transformer_reps)

        if len(reps) > 1:
            reps = numpy.concatenate(reps, axis=1)

        if self.apply_pca_on_stacked_reps:
            assert self.pca is not None and self.pca_is_trained, "a trained PCA model is needed for dimensionality reduction"
            reps = self.pca.transform(reps)

        return reps

    def prepare_autoencoder(self, text_list: List[str], tokens_list: List[List[str]], test_bundle: Tuple[List[str], List[List[str]]] = None, number_of_epochs: int = 200, batch_size: int = 128, shuffle: bool =True) -> Dict[str, List[float]]:
        """
        Parameters
        ----------
        text_list: `List[str]`, required
            The list of texts (please note that it is caller's responsibility to do any preprocessing beforehand).

        tokens_list: `List[List[str]]`, required
            The list of token lists

        number_of_epochs: `int`, optional (default=200)
            The number of epochs to train the provided auto-encoder module for.

        batch_size: `int`, optional (default=128)
            The batch-size for computationally heavier modules.

        shuffle: `bool`, optional (default=True)
            If provided, the representations will be shuffled.

        Returns
        -----------
        The list of epoch loss items
        """
        if self.autoencoder_is_trained:
            return

        optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=number_of_epochs)
        reps = {'train': self.get_stacked_representations(text_list=text_list, tokens_list=tokens_list)}

        if shuffle:
            numpy.random.shuffle(reps)

        if test_bundle is not None:
            modes_to_try = ['train', 'test']
            epoch_losses = {'train': [], 'test': []}
            reps['test'] = self.get_stacked_representations(text_list=test_bundle[0], tokens_list=test_bundle[1])
        else:
            modes_to_try = ['train']
            epoch_losses = {'train': []}

        epoch_index_range = tqdm(range(number_of_epochs))
        for epoch_index in epoch_index_range:
            for mode in modes_to_try:
                if mode == 'train':
                    self.autoencoder.train()
                elif mode == 'test':
                    self.autoencoder.eval()
                else:
                    raise ValueError

                batch_losses = []
                for i in range(len(reps[mode]) // batch_size + 1):
                    if mode == 'train':
                        optimizer.zero_grad()
                    start_index = batch_size * i
                    if start_index == len(reps[mode]):
                        break
                    end_index = batch_size * (i + 1)
                    if end_index > reps[mode].shape[0]:
                        end_index = reps[mode].shape[0]
                    x = torch.from_numpy(reps[mode][numpy.arange(start_index, end_index), :]).to(self.device).float()
                    loss = self.autoencoder(x, return_loss=True)
                    batch_losses.append(loss.item())
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_losses[mode].append(numpy.mean(batch_losses))
                epoch_index_range.set_description(f"Epoch: {epoch_index} [{mode}] / Loss: {epoch_losses[mode][-1]:.4f}")
                epoch_index_range.refresh()
            scheduler.step()

        self.autoencoder_is_trained = True
        self.autoencoder.eval()

        return epoch_losses

    def get_representations(self, text_list: List[str], batch_size=128, return_processed: bool = False) -> numpy.ndarray:
        """
        Parameters
        ----------
        text_list: `List[str]`, required
            The list of texts (preprocessing and tokenization will be performed on it).

        batch_size: `int`, optional (default=128)
            The batch-size for computationally heavier modules.

        Returns
        -----------
        The computed representations for the
        """
        preprocessed_text_list, preprocessed_tokens_list, indices = self.preprocess_and_get_text_and_tokens(
            text_list)

        reps = self.get_stacked_representations(text_list=preprocessed_text_list, tokens_list=preprocessed_tokens_list)

        if self.autoencoder is None:
            if return_processed:
                return reps, (preprocessed_text_list, preprocessed_tokens_list, indices)
            else:
                return reps
        else:
            assert self.autoencoder_is_trained

            embeddings = []
            for i in range(reps.shape[0] // batch_size + 1):
                start_index = batch_size * i
                if start_index == reps.shape[0]:
                    break
                end_index = batch_size * (i + 1)
                if end_index > reps.shape[0]:
                    end_index = reps.shape[0]
                x = torch.from_numpy(reps[numpy.arange(start_index, end_index), :]).to(self.device).float()
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                embeddings.append(self.autoencoder(x, return_embeddings=True))
            embeddings = torch.cat(embeddings, dim=0).data.cpu().numpy()

            if return_processed:
                return embeddings, (preprocessed_text_list, preprocessed_tokens_list, indices)
            else:
                return embeddings

    def train_clustering_fullbatch(self, reps: numpy.ndarray) -> None:
        """
        Parameters
        ----------
        reps: `numpy.ndarray`, required
            The representations to be clustered
        """
        self.representation_clustering.fit(reps)
        self.representation_clustering_is_trained = True

    def train_clustering_minibatch(self, reps: numpy.ndarray, batch_size: int = 128):
        """
        Parameters
        ----------
        reps: `numpy.ndarray`, required
            The representations to be clustered
        """
        for i in range(reps.shape[0] // batch_size + 1):
            start_index = batch_size * i
            if start_index == reps.shape[0]:
                break
            end_index = batch_size * (i + 1)
            if end_index > reps.shape[0]:
                end_index = reps.shape[0]
            self.representation_clustering.partial_fit(reps[numpy.arange(start_index, end_index), :])
        self.representation_clustering_is_trained = True

    def label_representation_cluster(self, reps):
        """
        Parameters
        ----------
        reps: `numpy.ndarray`, required
            The representations to be clustered

        Returns
        ----------
        The representations will be assigned a cluster labels, which will be returned by this method.
        """
        assert self.representation_clustering_is_trained, "clustering model is not trained yet"
        return self.representation_clustering.predict(reps)

    def train_pca(self, reps) -> None:
        """
        Parameters
        ----------
        reps: `numpy.ndarray`, required
            The representations to be clustered
        """
        self.pca.fit(reps)
        self.pca_is_trained = True
