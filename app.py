from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Only use SBERT model
SBERT_MODEL = None

def get_model():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        print("Loading SBERT model...")
        SBERT_MODEL = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return SBERT_MODEL

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/compare', methods=['POST'])
def compare_texts():
    try:
        data = request.json
        print(f"Received data: {data}")
        
        word = data.get('word', '')
        description = data.get('description', '')
        
        model = get_model()
        embeddings = model.encode([word, description])
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        response = jsonify({
            'similarity': float(similarity),
            'model': 'sbert'
        })
        print(f"Response: {response.get_json()}")
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
