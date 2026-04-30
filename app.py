from flask import Flask, render_template, request, jsonify
from summarizer import TextSummarizer
import traceback

app = Flask(__name__)
summarizer = TextSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        mode = data.get('mode', 'both')          # 'extractive', 'abstractive', 'both'
        ratio = float(data.get('ratio', 0.3))    # compression ratio 0.1–0.9

        if not text:
            return jsonify({'error': 'No text provided.'}), 400
        if len(text.split()) < 30:
            return jsonify({'error': 'Please enter at least 30 words for meaningful summarization.'}), 400

        result = {}

        if mode in ('extractive', 'both'):
            ext = summarizer.extractive_summarize(text, ratio)
            result['extractive'] = ext

        if mode in ('abstractive', 'both'):
            abst = summarizer.abstractive_summarize(text, ratio)
            result['abstractive'] = abst

        if mode == 'both':
            hybrid = summarizer.hybrid_summarize(text, ratio)
            result['hybrid'] = hybrid

        # Metrics
        original_words = len(text.split())
        result['original_words'] = original_words

        if 'extractive' in result:
            result['extractive_words'] = len(result['extractive'].split())
            result['extractive_reduction'] = round(
                (1 - result['extractive_words'] / original_words) * 100, 1)

        if 'abstractive' in result:
            result['abstractive_words'] = len(result['abstractive'].split())
            result['abstractive_reduction'] = round(
                (1 - result['abstractive_words'] / original_words) * 100, 1)

        if 'hybrid' in result:
            result['hybrid_words'] = len(result['hybrid'].split())
            result['hybrid_reduction'] = round(
                (1 - result['hybrid_words'] / original_words) * 100, 1)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
