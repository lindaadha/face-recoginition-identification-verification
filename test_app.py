import requests

resp = requests.post("http://127.0.0.1:5000/compare",
                     files={"file": open('test_face/ben_afflek1.jpg','rb'),
                     "file2": open('test_face/ben_afflek2.jpg','rb')})

print(resp.json())