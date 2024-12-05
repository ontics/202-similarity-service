from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import tensorflow as tf
# from word2vec import Word2VecHandler  # Comment this line out

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Load models lazily based on configuration
USE_MODEL = None
SBERT_MODEL = None
WORD2VEC_HANDLER = None

def get_model(model_choice):
    global USE_MODEL, SBERT_MODEL
    if model_choice == 'sbert':
        if SBERT_MODEL is None:
            print("Loading SBERT model...")
            SBERT_MODEL = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        return SBERT_MODEL
    else:
        if USE_MODEL is None:
            print("Loading USE model...")
            USE_MODEL = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        return USE_MODEL

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/compare', methods=['POST'])
def compare_texts():
    try:
        data = request.json
        print(f"Received data: {data}")
        
        word = data.get('word', '')
        description = data.get('description', '')
        model_choice = data.get('model', 'sbert')
        
        model = get_model(model_choice)
        
        if model_choice == 'sbert':
            # Get embeddings using Sentence-BERT
            embeddings = model.encode([word, description])
            similarity = cosine_similarity(embeddings[0], embeddings[1])
        else:
            # Get embeddings using Universal Sentence Encoder
            embeddings = model([word, description])
            similarity = cosine_similarity(embeddings[0].numpy(), embeddings[1].numpy())
        
        response = jsonify({
            'similarity': float(similarity),
            'model': model_choice
        })
        print(f"Response: {response.get_json()}")
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/vector-math', methods=['POST'])
def vector_math():
    try:
        global WORD2VEC_HANDLER
        if WORD2VEC_HANDLER is None:
            WORD2VEC_HANDLER = Word2VecHandler()
            
        data = request.json
        print(f"Received data: {data}")
        
        positive = data.get('positive', [])
        negative = data.get('negative', [])
        
        result = WORD2VEC_HANDLER.vector_math(positive, negative)
        
        if 'error' in result:
            return jsonify(result), 400
            
        response = jsonify(result)
        print(f"Response: {response.get_json()}")
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
