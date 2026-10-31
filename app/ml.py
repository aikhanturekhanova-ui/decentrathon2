"""Learned complaint-category model (optional, blended with the lexicon in classify.py).

Trained by `make train` (scripts/train_classifier.py) on TRAIN only, with weak labels derived from administrators' replies
(rule code such as «2.1 ОП», appeal verdicts). Features: TF-IDF of character n-grams (robust to slang, typos, transliteration
and the game's Kazakh look-alike letters) + word uni/bigrams; classifier: multinomial logistic regression.

The model file is built locally and is NOT committed: its vocabulary is derived from the confidential tickets.
Without it (or without scikit-learn) the service runs on the lexicon alone and /health says so.
Loading uses joblib (pickle): only load a file produced by `make train` on your own machine.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODEL_FORMAT = 2
_WORD = re.compile(r"(?u)\b\w\w+\b")  # the default token pattern of scikit-learn's word vectorizer


@dataclass
class ComplaintModel:
    pipeline: object  # sklearn Pipeline(vectorizer, classifier)
    classes: list[str]
    meta: dict = field(default_factory=dict)
    known_words: frozenset = frozenset()  # word unigrams seen in training: how much of a text the model has evidence for

    def coverage(self, text: str) -> float:
        """Share of the text's words seen in training (1.0 for an empty text). Low coverage (an unusual language or slang)
        means the probabilities rest on character fragments only, so classify.py trusts the lexicon more."""
        words = _WORD.findall(text.lower())
        if not words or not self.known_words:
            return 1.0
        return sum(w in self.known_words for w in words) / len(words)

    def proba(self, text: str) -> dict[str, float]:
        p = self.pipeline.predict_proba([text])[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def proba_many(self, texts: list[str]) -> list[dict[str, float]]:
        return [{c: float(v) for c, v in zip(self.classes, row)} for row in self.pipeline.predict_proba(texts)]

    # ---- persistence ----------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": MODEL_FORMAT, "pipeline": self.pipeline, "classes": self.classes, "meta": self.meta,
                     "known_words": sorted(self.known_words)}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> Optional["ComplaintModel"]:
        """None when the file is absent, scikit-learn is missing or the file is unreadable (logged to stderr, never fatal)."""
        if not Path(path).is_file():
            return None
        try:
            import joblib

            raw = joblib.load(path)
            if raw.get("format") != MODEL_FORMAT:
                raise ValueError(f"model format {raw.get('format')} != {MODEL_FORMAT}; run `make train` again")
            return cls(raw["pipeline"], list(raw["classes"]), dict(raw.get("meta") or {}), frozenset(raw.get("known_words") or ()))
        except Exception as exc:  # noqa: BLE001 - a broken optional model must not stop the service
            print(f"[WARNING] complaint model {path} not loaded ({type(exc).__name__}: {exc}); using the lexicon only", file=sys.stderr)
            return None


def build_pipeline(c: float = 4.0):
    """The configuration selected in the experiment documented in docs/metrics/ml_classifier.json."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    features = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))


def train(texts: list[str], labels: list[str], meta: Optional[dict] = None) -> ComplaintModel:
    pipe = build_pipeline()
    pipe.fit(texts, labels)
    word_vec = pipe.steps[0][1].transformer_list[1][1]
    known = frozenset(t for t in word_vec.vocabulary_ if " " not in t)
    return ComplaintModel(pipe, [str(c) for c in pipe.classes_], dict(meta or {}), known)
