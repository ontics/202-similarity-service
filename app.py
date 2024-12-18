from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time
import threading
import signal
import gc

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Only use SBERT model
SBERT_MODEL = None
STARTUP_TIME = time.time()
MODEL_LOAD_TIME = None
MODEL_LOCK = threading.Lock()
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received shutdown signal. Initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def get_model():
    global SBERT_MODEL, MODEL_LOAD_TIME
    if SBERT_MODEL is None:
        with MODEL_LOCK:
            if SBERT_MODEL is None:  # Double-check pattern
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading SBERT model...")
                start_time = time.time()
                SBERT_MODEL = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                MODEL_LOAD_TIME = time.time() - start_time
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SBERT model loaded in {MODEL_LOAD_TIME:.2f} seconds")
    return SBERT_MODEL

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/', methods=['GET', 'HEAD'])
def root():
    """Root endpoint for health checks."""
    return jsonify({
        "status": "ok",
        "message": "Similarity service is running"
    })

@app.route('/compare', methods=['POST'])
def compare_texts():
    if SHUTDOWN_REQUESTED:
        return jsonify({'error': 'Service is shutting down'}), 503

    try:
        data = request.json
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received comparison request: {data}")
        
        word = data.get('word', '')
        description = data.get('description', '')
        
        model = get_model()
        
        # Get embeddings for both texts at once
        embeddings = model.encode([word, description], batch_size=2, show_progress_bar=False)
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        # Force garbage collection after processing
        gc.collect()
        
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
    if SHUTDOWN_REQUESTED:
        return jsonify({'error': 'Service is shutting down'}), 503

    try:
        data = request.json
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Calculating batch similarity for {len(data)} pairs")
        
        model = get_model()
        
        # Prepare all texts in two lists
        words = [item['word'].lower() for item in data]
        descriptions = [item['description'].lower() for item in data]
        
        # Get embeddings for all texts at once
        start_time = time.time()
        word_embeddings = model.encode(words, batch_size=64, show_progress_bar=False)
        desc_embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=False)
        
        # Calculate similarities using vectorized operations
        # Normalize embeddings
        word_norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
        desc_norms = np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
        word_embeddings_normalized = word_embeddings / word_norms
        desc_embeddings_normalized = desc_embeddings / desc_norms
        
        # Calculate dot products
        similarities = np.sum(word_embeddings_normalized * desc_embeddings_normalized, axis=1)
        
        # Create results
        results = [
            {'similarity': float(sim), 'model': 'sbert'}
            for sim in similarities
        ]
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch processing completed in {time.time() - start_time:.2f} seconds")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify(results)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in batch comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if SHUTDOWN_REQUESTED:
        return jsonify({'error': 'Service is shutting down'}), 503

    try:
        # Try to get the model to ensure it's loaded
        model = get_model()
        
        uptime = time.time() - STARTUP_TIME
        model_status = "loaded" if model is not None else "not_loaded"
        
        # Do a quick test comparison to verify model is working
        test_embeddings = model.encode(["test", "test"], batch_size=2, show_progress_bar=False)
        test_similarity = cosine_similarity(test_embeddings[0], test_embeddings[1])
        
        health_data = {
            "status": "healthy",
            "uptime": f"{uptime:.2f} seconds",
            "model_status": model_status,
            "model_load_time": f"{MODEL_LOAD_TIME:.2f} seconds" if MODEL_LOAD_TIME else None,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "test_similarity": float(test_similarity)
        }
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Health check: {health_data}")
        return jsonify(health_data), 200
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting similarity service...")

# Preload the model on startup
if os.environ.get('PRELOAD_MODEL', 'true').lower() == 'true':
    get_model()

if __name__ == '__main__':
    # Get port from environment variable with fallback to 10000 (Render's default)
    port = int(os.environ.get('PORT', 10000))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting server on port {port}")
    app.run(host='0.0.0.0', port=5000)
