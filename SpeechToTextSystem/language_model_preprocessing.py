import nltk
from nltk.util import ngrams
from collections import Counter
import re

nltk.download('punkt')

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return tokens

def get_trigram_freq(tokens):
    trigrams = list(ngrams(tokens, 3))
    trigram_freq = Counter(trigrams)
    return trigram_freq

if __name__ == "__main__":
    sample_text = "Your text corpus goes here. This is a sample text for trigram frequency calculation."
    tokens = preprocess_text(sample_text)
    trigram_freq = get_trigram_freq(tokens)
    print("Most common trigrams:")
    for trigram, freq in trigram_freq.most_common(10):
        print(f"{trigram}: {freq}")
