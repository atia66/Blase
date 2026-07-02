import requests
import base64
import time
with open("./Chapter_3.ppt.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "file1": pdf_b64
}
start=time.time()
response = requests.post(
    "http://localhost:8000/pdf",
    json=payload
)
end=time.time()
print(response.status_code)
print(response.json())
print(f"{end-start:.3f} s")