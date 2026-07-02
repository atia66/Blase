import requests
import base64
import cv2

img = cv2.imread("./img/5915494917218176596.jpg") #5915494917218176596.jpg
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
_, buffer = cv2.imencode(".png", img)

img_b64 = base64.b64encode(buffer)

payload = {
    "bytes_img": img_b64
}

response = requests.post(
    "http://127.0.0.1:8000/request_bytes",
    json=payload
)

print(response.status_code)
print(response.text)
