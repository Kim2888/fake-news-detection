from flask import Flask, request, jsonify, render_template_string
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

model_path = 'saved_models/bert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; }
        h1 { color: #333; }
        textarea { width: 100%; height: 200px; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 6px; }
        button { margin-top: 12px; padding: 10px 30px; background: #4CAF50; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
        button:hover { background: #45a049; }
        #result { margin-top: 24px; font-size: 22px; font-weight: bold; }
        .fake { color: #e53935; }
        .real { color: #43a047; }
    </style>
</head>
<body>
    <h1>Fake News Detector</h1>
    <textarea id="news" placeholder="Paste news article text here..."></textarea>
    <br>
    <button onclick="predict()">Analyze</button>
    <div id="result"></div>

    <script>
        async function predict() {
            const text = document.getElementById('news').value.trim();
            if (!text) return;
            document.getElementById('result').innerHTML = 'Analyzing...';
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            const label = data.label;
            const conf = (data.confidence * 100).toFixed(1);
            const cls = label === 'FAKE' ? 'fake' : 'real';
            document.getElementById('result').innerHTML =
                `<span class="${cls}">${label}</span> &nbsp; (${conf}% confidence)`;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    label = 'REAL' if pred == 1 else 'FAKE'
    confidence = probs[pred].item()
    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
