from flask import Flask, request, jsonify, render_template
import joblib
import nltk
import re
import os

nltk.download('punkt')

# Load models
vectorizer = joblib.load('tfidf_vectorizer.pkl')
clf_condition = joblib.load('condition_model.pkl')
clf_risk = joblib.load('risk_model.pkl')
clf_sentiment = joblib.load('sentiment_model.pkl')

le_condition = joblib.load('le_condition.pkl')
le_risk = joblib.load('le_risk.pkl')
le_sentiment = joblib.load('le_sentiment.pkl')

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        raw_text = request.form['text']
        cleaned = clean_text(raw_text)
        tfidf_text = vectorizer.transform([cleaned])

        condition_pred = clf_condition.predict(tfidf_text)
        risk_pred = clf_risk.predict(tfidf_text)
        sentiment_pred = clf_sentiment.predict(tfidf_text)

        result = {
            'condition': le_condition.inverse_transform(condition_pred)[0],
            'risk': le_risk.inverse_transform(risk_pred)[0],
            'sentiment': le_sentiment.inverse_transform(sentiment_pred)[0]
        }

    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    raw_text = data.get('text', '')

    if not raw_text or not isinstance(raw_text, str):
        return jsonify({'error': 'Invalid or empty input'}), 400

    cleaned_text = clean_text(raw_text)
    tfidf_text = vectorizer.transform([cleaned_text])

    condition_pred = clf_condition.predict(tfidf_text)
    risk_pred = clf_risk.predict(tfidf_text)
    sentiment_pred = clf_sentiment.predict(tfidf_text)

    response = {
        'text': raw_text,
        'predicted_condition': le_condition.inverse_transform(condition_pred)[0],
        'predicted_risk_level': le_risk.inverse_transform(risk_pred)[0],
        'predicted_sentiment': le_sentiment.inverse_transform(sentiment_pred)[0]
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
