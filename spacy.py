import spacy
from spacy.matcher import Matcher
from textblob import TextBlob

# Load spaCy Model
nlp = spacy.load("en_core_web_sm")

# 📌 1. Tokenization & Lemmatization
def process_text(text):
    """Tokenizes text and performs lemmatization"""
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

# 📌 2. Named Entity Recognition (NER)
def extract_entities(text):
    """Extracts named entities from text"""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# 📌 3. Rule-Based Matching (Keyword Extraction)
def keyword_matcher(text, pattern_word):
    """Finds specific words in text using SpaCy Matcher"""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": pattern_word.lower()}]
    matcher.add("KeywordPattern", [pattern])
    matches = matcher(doc)
    return [doc[start:end].text for match_id, start, end in matches]

# 📌 4. Text Similarity
def check_similarity(text1, text2):
    """Compares two texts for similarity"""
    return nlp(text1).similarity(nlp(text2))

# 📌 5. Sentiment Analysis
def get_sentiment(text):
    """Returns sentiment polarity score (-1 to 1)"""
    return TextBlob(text).sentiment.polarity

# ================== Example Usage ==================
text = "Apple Inc. was founded by Steve Jobs in 1976."
text2 = "Apple was started by Steve Jobs."

# Tokenization & Lemmatization
print("🔹 Tokenized & Lemmatized Text:", process_text(text))

# Named Entity Recognition (NER)
print("🔹 Named Entities:", extract_entities(text))

# Rule-Based Matching
print("🔹 Keyword Match (Apple):", keyword_matcher(text, "Apple"))

# Text Similarity
print("🔹 Similarity Score:", check_similarity(text, text2))

# Sentiment Analysis
print("🔹 Sentiment Score:", get_sentiment("I love this product!"))
