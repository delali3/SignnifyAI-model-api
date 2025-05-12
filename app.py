import requests

url = 'https://signnifyai-model-api.onrender.com/predict'
files = {'file': open('download.jpeg', 'rb')}
response = requests.post(url, files=files)

print(response.json())
