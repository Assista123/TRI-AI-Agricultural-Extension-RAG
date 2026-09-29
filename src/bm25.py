"""Pure-Python BM25 Okapi implementation."""

import math
from collections import Counter


class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = (
            sum(len(doc) for doc in corpus) / self.corpus_size
            if self.corpus_size > 0
            else 0
        )
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        df = Counter()
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] += 1

        for word, freq in df.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )

    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for i, doc_freqs in enumerate(self.doc_freqs):
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    numerator = idf * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (
                        1 - self.b + self.b * (self.doc_len[i] / self.avgdl)
                    )
                    scores[i] += numerator / denominator
        return scores
