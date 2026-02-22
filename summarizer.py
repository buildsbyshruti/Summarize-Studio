"""
TextSummarizer
==============
Implements both extractive (TextRank + TF-IDF) and abstractive
(facebook/bart-large-cnn via HuggingFace Transformers) summarization.
"""

import re
import math
import heapq
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data silently
for pkg in ('punkt', 'stopwords', 'punkt_tab'):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class TextSummarizer:
    """Handles both extractive and abstractive text summarization."""

    def __init__(self):
        self._stop_words = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()
        self._abst_pipeline = None   # Lazy-load transformer pipeline

    # ------------------------------------------------------------------ #
    #  EXTRACTIVE  –  TextRank with TF-IDF sentence scoring               #
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tokenize_words(self, sentence: str):
        words = word_tokenize(sentence.lower())
        return [
            self._stemmer.stem(w) for w in words
            if w.isalnum() and w not in self._stop_words
        ]

    def _tfidf_scores(self, sentences):
        """Compute TF-IDF based importance score for each sentence."""
        tf = []
        df = defaultdict(int)
        N = len(sentences)

        for sent in sentences:
            words = self._tokenize_words(sent)
            freq = defaultdict(int)
            for w in words:
                freq[w] += 1
            tf.append(freq)
            for w in freq:
                df[w] += 1

        scores = []
        for freq in tf:
            score = 0.0
            total = sum(freq.values()) or 1
            for w, cnt in freq.items():
                tf_val = cnt / total
                idf_val = math.log((N + 1) / (df[w] + 1)) + 1
                score += tf_val * idf_val
            scores.append(score)

        return scores

    def _textrank_scores(self, sentences, damping=0.85, iterations=30):
        """PageRank over sentence similarity graph."""
        n = len(sentences)
        if n == 0:
            return []

        # Build word-set per sentence
        word_sets = [set(self._tokenize_words(s)) for s in sentences]

        # Similarity matrix (cosine-like Jaccard)
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                union = word_sets[i] | word_sets[j]
                if not union:
                    continue
                sim[i][j] = len(word_sets[i] & word_sets[j]) / math.sqrt(
                    len(word_sets[i]) * len(word_sets[j]) + 1e-9)

        # Normalize rows
        for i in range(n):
            row_sum = sum(sim[i]) or 1
            sim[i] = [v / row_sum for v in sim[i]]

        # Power iteration
        rank = [1.0 / n] * n
        for _ in range(iterations):
            new_rank = []
            for i in range(n):
                s = sum(sim[j][i] * rank[j] for j in range(n))
                new_rank.append((1 - damping) / n + damping * s)
            rank = new_rank

        return rank

    def extractive_summarize(self, text: str, ratio: float = 0.3) -> str:
        """Return extractive summary using combined TF-IDF + TextRank scoring."""
        text = self._clean(text)
        sentences = sent_tokenize(text)

        if len(sentences) <= 3:
            return text

        n_summary = max(1, round(len(sentences) * ratio))

        tfidf = self._tfidf_scores(sentences)
        tr = self._textrank_scores(sentences)

        # Normalize and combine
        max_tfidf = max(tfidf) or 1
        max_tr = max(tr) or 1
        combined = [
            0.5 * (tfidf[i] / max_tfidf) + 0.5 * (tr[i] / max_tr)
            for i in range(len(sentences))
        ]

        # Pick top-N sentences, preserving original order
        top_indices = set(heapq.nlargest(
            n_summary, range(len(combined)), key=lambda i: combined[i]))

        summary = ' '.join(
            sentences[i] for i in sorted(top_indices)
        )
        return summary

    # ------------------------------------------------------------------ #
    #  ABSTRACTIVE  –  facebook/bart-large-cnn (HuggingFace)              #
    # ------------------------------------------------------------------ #

    def _load_abstractive_pipeline(self):
        if self._abst_pipeline is not None:
            return
        from transformers import pipeline
        # distilbart-cnn-6-6: ~306 MB, ~3× faster than bart-large-cnn on CPU
        self._abst_pipeline = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-6-6',
            tokenizer='sshleifer/distilbart-cnn-6-6',
            device=-1      # CPU; set 0 for CUDA GPU
        )

    def abstractive_summarize(self, text: str, ratio: float = 0.3) -> str:
        """Return abstractive summary using distilbart-cnn-6-6 (fast CPU model)."""
        self._load_abstractive_pipeline()

        original_words = len(text.split())
        max_len = max(30, int(original_words * ratio))
        min_len = max(20, int(max_len * 0.5))

        # distilbart accepts up to ~1024 tokens; use smaller chunks for speed
        MAX_WORDS = 400
        words = text.split()

        if len(words) <= MAX_WORDS:
            chunks = [text]
        else:
            # Split into ~MAX_WORDS word chunks with slight overlap
            chunks = []
            step = MAX_WORDS - 50
            for start in range(0, len(words), step):
                chunk = ' '.join(words[start: start + MAX_WORDS])
                chunks.append(chunk)

        summaries = []
        chunk_max = max(30, max_len // len(chunks))
        chunk_min = max(15, min_len // len(chunks))

        for chunk in chunks:
            out = self._abst_pipeline(
                chunk,
                max_length=chunk_max,
                min_length=chunk_min,
                do_sample=False,
                truncation=True,
            )
            summaries.append(out[0]['summary_text'])

        # If multiple chunks, do a second-pass summary
        combined = ' '.join(summaries)
        if len(chunks) > 1 and len(combined.split()) > max_len * 1.5:
            out = self._abst_pipeline(
                combined,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            return out[0]['summary_text']

        return combined
