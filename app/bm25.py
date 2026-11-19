import math
import re
from collections import Counter
from typing import List, Tuple

def default_tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t]

class BM25Okapi:
    """
    Pure-Python, zero-dependency, high-performance implementation of BM25Okapi.
    Formula:
      IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
      Score(D, Q) = sum_{q in Q} IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    """
    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.doc_lengths = []
        self.doc_freqs = Counter()
        self.doc_token_counts = []

        total_length = 0
        for text in corpus_texts:
            tokens = default_tokenize(text)
            counts = Counter(tokens)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length
            self.doc_token_counts.append(counts)
            for token in counts.keys():
                self.doc_freqs[token] += 1

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Precompute IDFs
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> List[float]:
        q_tokens = default_tokenize(query)
        scores = [0.0] * self.corpus_size
        if not q_tokens or self.corpus_size == 0:
            return scores

        for q in q_tokens:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]
            for doc_idx in range(self.corpus_size):
                f = self.doc_token_counts[doc_idx].get(q, 0)
                if f > 0:
                    d_len = self.doc_lengths[doc_idx]
                    denom = f + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl))
                    scores[doc_idx] += idf_val * (f * (self.k1 + 1.0)) / denom

        return scores

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
