import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# Step 1: Load dataset
dataset_path = "dataset.xlsx"  # Replace with your file path
data = pd.read_excel(dataset_path)

print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Step 2: Clean dataset
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data = data.dropna(subset=['text', 'label'])
data = data[data['label'].isin([0, 1])]
data = data.drop_duplicates()
data['text'] = data['text'].apply(clean_text)
data = data[data['text'].str.len().between(5, 500)]
data = data[~data['text'].str.contains('errorname|rt dbae')]

print("\nCleaned Dataset Info:")
print(data.info())
print("\nLabel Distribution:")
print(data['label'].value_counts())

# Step 3: Preprocess dataset
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

data['text'] = data['text'].apply(tokenize_text)

X = data['text']
y = data['label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nData Split Sizes:")
print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
print("\nTraining Label Distribution:")
print(y_train.value_counts())

# Step 4: Feature extraction
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nTF-IDF Feature Shape:")
print(f"Training: {X_train_tfidf.shape}, Validation: {X_val_tfidf.shape}, Test: {X_test_tfidf.shape}")

# Step 5: Rule-based labeling
sid = SentimentIntensityAnalyzer()
academic_keywords = [
    'exam', 'thesis', 'study', 'deadline', 'lecture', 'assignment', 'quiz',
    'presentation', 'class', 'school', 'university', 'course', 'grade', 'semester'
]

def detect_academic_stress(text, label, sentiment):
    text_lower = text.lower()
    has_academic_keyword = any(keyword in text_lower for keyword in academic_keywords)
    if has_academic_keyword:
        if label == 1:
            return 'Academic Stress'
        elif label == 0 and sentiment in ['Neutral', 'Negative']:
            return 'Academic Stress'
    return None

def assign_condition(row):
    sentiment = get_sentiment(row['text'])
    academic_stress = detect_academic_stress(row['text'], row['label'], sentiment)
    if academic_stress:
        return academic_stress
    elif row['label'] == 1:
        return 'Depression'
    else:
        return 'Normal Anxiety'

def get_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

data['sentiment'] = data['text'].apply(get_sentiment)
data['condition'] = data.apply(assign_condition, axis=1)

risk_mapping = {
    'Depression': 'High',
    'Academic Stress': 'Medium',
    'Normal Anxiety': 'Low'
}
data['risk_level'] = data['condition'].map(risk_mapping)

le_sentiment = LabelEncoder()
data['sentiment_encoded'] = le_sentiment.fit_transform(data['sentiment'])
le_condition = LabelEncoder()
data['condition_encoded'] = le_condition.fit_transform(data['condition'])
le_risk = LabelEncoder()
data['risk_encoded'] = le_risk.fit_transform(data['risk_level'])

joblib.dump(le_sentiment, 'le_sentiment.pkl')
joblib.dump(le_condition, 'le_condition.pkl')
joblib.dump(le_risk, 'le_risk.pkl')

print("\nCondition Distribution:")
print(data['condition'].value_counts())
print("\nRisk Level Distribution:")
print(data['risk_level'].value_counts())
print("\nSentiment Distribution:")
print(data['sentiment'].value_counts())

# Step 6: Train models
# Condition model
clf_condition = LogisticRegression(max_iter=1000, class_weight='balanced')
y_condition = data['condition_encoded']
X_train_cond, X_temp_cond, y_train_cond, y_temp_cond = train_test_split(X, y_condition, test_size=0.2, random_state=42)
X_val_cond, X_test_cond, y_val_cond, y_test_cond = train_test_split(X_temp_cond, y_temp_cond, test_size=0.5, random_state=42)
X_train_cond_tfidf = vectorizer.transform(X_train_cond)
X_val_cond_tfidf = vectorizer.transform(X_val_cond)
X_test_cond_tfidf = vectorizer.transform(X_test_cond)
clf_condition.fit(X_train_cond_tfidf, y_train_cond)

print("\nCondition Model - Validation Set Performance:")
print(classification_report(y_val_cond, clf_condition.predict(X_val_cond_tfidf), target_names=le_condition.classes_))
print("\nCondition Model - Test Set Performance:")
print(classification_report(y_test_cond, clf_condition.predict(X_test_cond_tfidf), target_names=le_condition.classes_))

# Risk level model
clf_risk = LogisticRegression(max_iter=1000, class_weight='balanced')
y_risk = data['risk_encoded']
X_train_risk, X_temp_risk, y_train_risk, y_temp_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)
X_val_risk, X_test_risk, y_val_risk, y_test_risk = train_test_split(X_temp_risk, y_temp_risk, test_size=0.5, random_state=42)
X_train_risk_tfidf = vectorizer.transform(X_train_risk)
X_val_risk_tfidf = vectorizer.transform(X_val_risk)
X_test_risk_tfidf = vectorizer.transform(X_test_risk)
clf_risk.fit(X_train_risk_tfidf, y_train_risk)

print("\nRisk Level Model - Validation Set Performance:")
print(classification_report(y_val_risk, clf_risk.predict(X_val_risk_tfidf), target_names=le_risk.classes_))
print("\nRisk Level Model - Test Set Performance:")
print(classification_report(y_test_risk, clf_risk.predict(X_test_risk_tfidf), target_names=le_risk.classes_))

# Sentiment model
clf_sentiment = LogisticRegression(max_iter=1000, class_weight='balanced')
y_sentiment = data['sentiment_encoded']
X_train_sent, X_temp_sent, y_train_sent, y_temp_sent = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)
X_val_sent, X_test_sent, y_val_sent, y_test_sent = train_test_split(X_temp_sent, y_temp_sent, test_size=0.5, random_state=42)
X_train_sent_tfidf = vectorizer.transform(X_train_sent)
X_val_sent_tfidf = vectorizer.transform(X_val_sent)
X_test_sent_tfidf = vectorizer.transform(X_test_sent)
clf_sentiment.fit(X_train_sent_tfidf, y_train_sent)

print("\nSentiment Model - Validation Set Performance:")
print(classification_report(y_val_sent, clf_sentiment.predict(X_val_sent_tfidf), target_names=le_sentiment.classes_))
print("\nSentiment Model - Test Set Performance:")
print(classification_report(y_test_sent, clf_sentiment.predict(X_test_sent_tfidf), target_names=le_sentiment.classes_))

joblib.dump(clf_condition, 'condition_model.pkl')
joblib.dump(clf_risk, 'risk_model.pkl')
joblib.dump(clf_sentiment, 'sentiment_model.pkl')

# Step 7: Test inference
sample_text = ["i am blessed but i feel a little stressed"]
sample_text_clean = [clean_text(text) for text in sample_text]
sample_text_tokenized = [tokenize_text(text) for text in sample_text_clean]
sample_tfidf = vectorizer.transform(sample_text_tokenized)

condition_pred = clf_condition.predict(sample_tfidf)
risk_pred = clf_risk.predict(sample_tfidf)
sentiment_pred = clf_sentiment.predict(sample_tfidf)

condition_label = le_condition.inverse_transform(condition_pred)[0]
risk_label = le_risk.inverse_transform(risk_pred)[0]
sentiment_label = le_sentiment.inverse_transform(sentiment_pred)[0]

print("\nSample Prediction:")
print(f"Text: {sample_text[0]}")
print(f"Predicted Condition: {condition_label}")
print(f"Predicted Risk Level: {risk_label}")
print(f"Predicted Sentiment: {sentiment_label}")
