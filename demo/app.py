import anago
from anago.utils import download
from flask import Flask, render_template, jsonify, request
url = 'https://storage.googleapis.com/chakki/datasets/public/ner/conll2003_en.zip'
weights, params, preprocessor = download(url)
model = anago.Sequence.load(weights, params, preprocessor)
print(model.analyze('Obama'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ner', methods=['POST'])
def analyzer():
    j = request.get_json()
    res = model.analyze(j.get('text'))
    return jsonify(res)
