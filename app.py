from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Only use SBERT model
SBERT_MODEL = None
STARTUP_TIME = time.time()
MODEL_LOAD_TIME = None

def get_model():
    global SBERT_MODEL, MODEL_LOAD_TIME
    if SBERT_MODEL is None:
        print("Loading SBERT model...")
        start_time = time.time()
        SBERT_MODEL = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        MODEL_LOAD_TIME = time.time() - start_time
        print(f"SBERT model loaded in {MODEL_LOAD_TIME:.2f} seconds")
    return SBERT_MODEL

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/compare', methods=['POST'])
def compare_texts():
    try:
        data = request.json
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received comparison request: {data}")
        
        word = data.get('word', '')
        description = data.get('description', '')
        
        model = get_model()
        embeddings = model.encode([word, description])
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        response = jsonify({
            'similarity': float(similarity),
            'model': 'sbert'
        })
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparison response: {response.get_json()}")
        return response
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    uptime = time.time() - STARTUP_TIME
    model_status = "loaded" if SBERT_MODEL is not None else "not_loaded"
    health_data = {
        "status": "healthy",
        "uptime": f"{uptime:.2f} seconds",
        "model_status": model_status,
        "model_load_time": f"{MODEL_LOAD_TIME:.2f} seconds" if MODEL_LOAD_TIME else None,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Health check: {health_data}")
    return jsonify(health_data), 200

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting similarity service...")
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
