from typing import Iterable, Union
import unicodedata
import re


def normalize_unicode(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


class Preprocess:
    def __init__(
        self, remove_punct: bool = True, stop_words: bool = [], normalize: bool = True
    ) -> None:
        self.remove_punct = remove_punct
        self.stop_words = stop_words
        self.normalize = normalize

    def fit(self, X: Iterable, y: Union[Iterable, None] = None):
        return self

    def transform(self, X: Iterable, y: Union[Iterable, None] = None) -> Iterable:
        if self.remove_punct:
            X = [
                re.sub(
                    "[\ʼ\!\"\#\$\%\&\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\\\]\^\_\`\{\|\}\~]",
                    "",
                    text.lower(),
                )
                for text in X
            ]
        X = [re.sub("[\n\s+]", " ", text.lower()) for text in X]
        if self.normalize:
            X = [normalize_unicode(text) for text in X]

        for stop_word in self.stop_words:
            X = [re.sub(f"^{stop_word}$", text, re.I | re.M) for text in X]
        return X
