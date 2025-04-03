import spacy
from spacy.matcher import Matcher
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# 📌 1️⃣ Tokenization & Lemmatization
# Definition: Tokenization breaks text into words or phrases, while lemmatization converts words to their base form.
# Use Case: Used in text preprocessing for search engines, chatbots, and AI writing assistants.
def process_text(text):
    """Tokenizes text and performs lemmatization"""
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

# 📌 2️⃣ Named Entity Recognition (NER)
# Definition: Identifies entities such as people, places, organizations, and dates in text.
# Use Case: Used in news extraction, finance, and AI chatbots to identify relevant data.
def extract_entities(text):
    """Extracts named entities from text"""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# 📌 3️⃣ Rule-Based Matching (Keyword Extraction)
# Definition: Uses pattern-based rules to find words or phrases in text.
# Use Case: Used in spam detection, chatbot responses, and content filtering.
def keyword_matcher(text, pattern_word):
    """Finds specific words in text using spaCy Matcher"""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": pattern_word.lower()}]
    matcher.add("KeywordPattern", [pattern])
    matches = matcher(doc)
    return [doc[start:end].text for match_id, start, end in matches]

# 📌 4️⃣ Text Similarity
# Definition: Measures how similar two pieces of text are.
# Use Case: Used in plagiarism detection, recommendation systems, and search engines.
def check_similarity(text1, text2):
    """Compares two texts for similarity"""
    return nlp(text1).similarity(nlp(text2))

# 📌 5️⃣ Sentiment Analysis
# Definition: Determines whether text expresses a **positive, negative, or neutral** sentiment.
# Use Case: Used in social media monitoring, product reviews, and customer feedback analysis.
def get_sentiment(text):
    """Returns sentiment polarity score (-1 to 1)"""
    return TextBlob(text).sentiment.polarity

# ================== Example Usage ==================
text = "Apple Inc. was founded by Steve Jobs in 1976."
text2 = "Apple was started by Steve Jobs."

# Tokenization & Lemmatization
print("🔹 Tokenized & Lemmatized:", process_text(text))

# Named Entity Recognition (NER)
print("🔹 Named Entities:", extract_entities(text))

# Rule-Based Matching
print("🔹 Keyword Match (Apple):", keyword_matcher(text, "Apple"))

# Text Similarity
print("🔹 Similarity Score:", check_similarity(text, text2))

# Sentiment Analysis
print("🔹 Sentiment Score:", get_sentiment("I love this product!"))
