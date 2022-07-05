import re

import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing


class NgramSegmenter:
    def __init__(
        self,
        predictor: sklearn.pipeline.Pipeline,
        label_encoder: preprocessing.LabelEncoder,
    ):
        """
        Segment text by language.
        :param predictor: sklearn pipeline that takes a text and produces
            predictions for the language class. The predictor must support
            `predict_proba`.
        :param label_encoder: sklearn label encoder for the language classes
        """
        self.predictor = predictor
        self.label_encoder = label_encoder

    def _get_ngram_indices(self, length: int, ngram: int = 3) -> np.array:
        """
        Return array of indices for each ngram sequence.
        :param length: Length of text.
        :param ngram: The ngram to extract
        """
        ngrams = []
        for i in range(0, length - ngram):
            ngrams.append(list(range(i, i + ngram)))
        return np.array(ngrams)

    def _compute_ngram_lang(
        self, text_array: np.array, ngram_indices: np.array, scores_array: np.array
    ) -> np.array:
        """
        Compute the likelihood of each token token belonging to each
        class using the predictor.
        For each ngram sequence string, the probability  for each class is computed
        and added to the likelihood score array for each index in the said ngram.
        :param text_array: array of text to be segmented. shape `(1, token_length)`
        :param ngram_indices: array if indices for the text's ngram. shape `(len(text_array)-ngram, ngram)`
        :param score_array: array of class scores for each token in text_array.
            shape `(len(text_array), classes_length)`
        """
        for indices in ngram_indices:
            text = " ".join(text_array[indices])
            probs = self.predictor.predict_proba([text])[0]
            scores_array[indices] += probs
        return scores_array

    def _smooth(self, spans: list) -> list:
        """
        Smoothen the assigned classes span.
        Spans for a class with length less than `k` are reassigned
        to the previous class.
        :param spans: list of spans of form `((start, end), class)`
        :param k: threshold length for spans. Any span with length
            less than this is reassigned to the preceeding span's class.
        """
        new_spans = []
        prev_span = spans[0][1]
        for i, span in enumerate(spans):
            sp = span[0]
            duplicate = False
            if new_spans and span:
                duplicate = span[1] == new_spans[len(new_spans) - 1][1]
            if ((sp[1] - sp[0] < self.k) and i > 0) or duplicate:
                span = ((new_spans[-1][0][0], sp[1]), prev_span)
                new_spans.pop()

            prev_span = span[1]
            new_spans.append(span)
        return new_spans

    def _get_spans(self, matches: list) -> tuple:
        """
        Extract spans from classes assigned to individual tokens.
        :param matches: list of class, token pairs for each token in text.
        """
        indexes = []
        prev = None
        index = []
        for i, match in enumerate(matches):
            if match[0] != prev:
                indexes.append((index, prev))
                index = []
            index.append(i)
            prev = match[0]
        spans = (
            pd.Series(indexes[1:])
            .apply(lambda x: ((x[0][0], x[0][-1] + 1), x[-1]))
            .values
        )
        # Smoothen the span and matches
        spans = self._smooth(spans)
        for span in spans:
            for i in range(*span[0]):
                matches[i] = (span[1], matches[i][1])
        return matches, spans

    def extract(self, text: str, ngram: int = 6, k: int = 4) -> tuple:
        """
        Segment text by supported languages.

        Note: Multiple consecutive spaces are collapsed into a
        single space

        :param text: text to be segmented.
        :param ngram: ngram to use in text segmentation.
            The default value (5) is pretty accurate for many tested texts.
        :param k: threshold length for spans. Any span with length
            less than this is reassigned to the preceeding span's class.
        """
        self.k = k
        text = re.sub(" +", " ", text)
        text = re.sub("(?<!\s)\.", " .", text)
        text = re.sub("\.(?!=\s)", ". ", text)
        text += " ."
        text_array = np.array([i for i in text.split(" ") if i])
        ngram_indices = self._get_ngram_indices(text_array.shape[0], ngram)
        scores_array = np.zeros(
            (text_array.shape[0], self.label_encoder.classes_.shape[0])
        )
        langs = self._compute_ngram_lang(text_array, ngram_indices, scores_array)
        language_indices = np.argmax(langs, axis=1)
        matches = list(
            zip(self.label_encoder.inverse_transform(language_indices), text_array)
        )
        matches, spans = self._get_spans(matches)
        return text_array[:-1], matches[:-1], spans
