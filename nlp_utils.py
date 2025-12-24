# nlp_utils.py
import re
from collections import Counter

STOPWORDS = {
    "for", "and", "the", "with", "of", "to", "a", "an", "in",
    "is", "are", "on", "or", "as", "by", "use", "used", "can",
    "helps", "help", "may", "also", "from", "into", "this", "that"
}

def tokenize(text):
    # lowercase + sirf letters
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def extract_keywords(text, top_k=5):
    """
    Very simple frequency-based keyword extractor.
    MSc level ke liye enough + explain karna easy.
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    # sabse frequent top_k words
    return [w for w, _ in counts.most_common(top_k)]

