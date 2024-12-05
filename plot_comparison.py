import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

# Example test cases
test_cases = [
    ("dog", "A golden retriever playing fetch in a sunny park"),
    ("car", "A red sports car speeding down an empty highway"),
    ("ocean", "A golden retriever playing fetch in a sunny park"),
    ("computer", "A red sports car speeding down an empty highway"),
    ("automobile", "A red sports car speeding down an empty highway")
]

use_scores = []
sbert_scores = []
use_times = []
sbert_times = []

# Base URL for the similarity service
base_url = "http://127.0.0.1:5000/compare"

# Function to get similarity score and response time
def get_similarity(word, description, model):
    response = requests.post(base_url, json={"word": word, "description": description, "model": model})
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    try:
        data = response.json()
        return data['similarity'], response.elapsed.total_seconds()
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON:", response.text)
        return None, None

# Collect scores and times for each test case
for word, description in test_cases:
    similarity, time_taken = get_similarity(word, description, "use")
    use_scores.append(similarity)
    use_times.append(time_taken)

    similarity, time_taken = get_similarity(word, description, "sbert")
    sbert_scores.append(similarity)
    sbert_times.append(time_taken)

# Plot similarity scores
plt.figure(figsize=(10, 5))
sns.lineplot(x=[tc[0] for tc in test_cases], y=use_scores, marker='o', label='USE')
sns.lineplot(x=[tc[0] for tc in test_cases], y=sbert_scores, marker='o', label='SBERT')
plt.title('Similarity Scores Comparison')
plt.ylabel('Similarity Score')
plt.xlabel('Test Case')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Plot response times
plt.figure(figsize=(10, 5))
sns.lineplot(x=[tc[0] for tc in test_cases], y=use_times, marker='o', label='USE')
sns.lineplot(x=[tc[0] for tc in test_cases], y=sbert_times, marker='o', label='SBERT')
plt.title('Response Time Comparison')
plt.ylabel('Response Time (seconds)')
plt.xlabel('Test Case')
plt.legend()
plt.show()