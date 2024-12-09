from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Only use SBERT model
SBERT_MODEL = None
STARTUP_TIME = time.time()
MODEL_LOAD_TIME = None
MODEL_LOCK = threading.Lock()

def get_model():
    global SBERT_MODEL, MODEL_LOAD_TIME
    if SBERT_MODEL is None:
        with MODEL_LOCK:
            if SBERT_MODEL is None:  # Double-check pattern
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
        
        # Get embeddings for both texts at once
        embeddings = model.encode([word, description], batch_size=2, show_progress_bar=False)
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

@app.route('/compare-batch', methods=['POST'])
def compare_texts_batch():
    try:
        data = request.json
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received batch comparison request with {len(data)} items")
        
        model = get_model()
        
        # Prepare all texts for batch encoding
        all_texts = []
        for item in data:
            all_texts.extend([item['word'], item['description']])
        
        # Get embeddings for all texts at once
        embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=False)
        
        # Calculate similarities
        results = []
        for i in range(0, len(embeddings), 2):
            similarity = cosine_similarity(embeddings[i], embeddings[i+1])
            results.append({
                'similarity': float(similarity),
                'model': 'sbert'
            })
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch comparison complete")
        return jsonify(results)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in batch comparison: {str(e)}")
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

# Preload the model on startup
if os.environ.get('PRELOAD_MODEL', 'true').lower() == 'true':
    get_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
