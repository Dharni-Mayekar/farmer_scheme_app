
import os, json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model and encoders
MODEL_DIR = os.path.join(os.getcwd(), 'model')

model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.joblib'))
encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.joblib'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.joblib'))

# Load bilingual scheme data
with open(os.path.join(os.path.dirname(__file__), 'bilingual_schemes.json'), encoding='utf-8') as f:
    SCHEMES = json.load(f)

CAT_COLS = ['land_size', 'crop_type', 'district', 'irrigation', 'farming_type']


def encode_input(data):
    row = []
    row.append(float(data['age']))
    row.append(float(data['income_lpa']))
    for col in CAT_COLS:
        le = encoders[col]
        val = data[col]
        if val in le.classes_:
            row.append(le.transform([val])[0])
        else:
            # fallback to most common class
            row.append(0)
    return np.array(row).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        X = encode_input(data)
        preds = model.predict(X)[0]  # shape (25,)
        probas = model.predict_proba(X)  # list of 25 arrays

        eligible = []
        for i, pred in enumerate(preds):
            scheme_id = i + 1
            if pred == 1:
                proba = probas[i][0][1] if hasattr(probas[i][0], '__len__') else probas[i][1]
                s = SCHEMES.get(str(scheme_id), {})
                eligible.append({
                    'scheme_id': scheme_id,
                    'confidence': round(float(proba) * 100, 1),
                    'name_en': s.get('name_en', f'Scheme {scheme_id}'),
                    'name_mr': s.get('name_mr', ''),
                    'benefit_en': s.get('benefit_en', ''),
                    'benefit_mr': s.get('benefit_mr', ''),
                    'documents': s.get('documents', []),
                    'apply_online': s.get('apply_online', ''),
                    'apply_offline': s.get('apply_offline', ''),
                    'category': s.get('category', ''),
                })

        eligible.sort(key=lambda x: x['confidence'], reverse=True)
        return jsonify({'count': len(eligible), 'schemes': eligible, 'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False)
