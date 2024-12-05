import requests

try:
    response = requests.post("http://localhost:5000/compare", json={"word": "test", "description": "test", "model": "use"})
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
except Exception as e:
    print("Error:", str(e))