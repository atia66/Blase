from fastapi import FastAPI ,UploadFile, File

from pydantic import BaseModel
import base64
import cv2
import numpy as np
from Machine.producer import request_producer
import uvicorn
app = FastAPI(
    title="scanner machine simulation",
    version="1.0.0")


class RequestModel(BaseModel):
    bytes_img: str

def image_encode(image):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def image_decode(b64_string):

    try:
        img_bytes = base64.b64decode(b64_string)

        img = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        return img

    except Exception as e:
        print("decode error:", e)
        return None

@app.post("/request_bytes")
async def detect(request: RequestModel):
    
    request_producer(request.bytes_img)
@app.post("/request_image")

async def detect_image(file: UploadFile = File(...)):

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "invalid image"}

    encoded_img = image_encode(image)

    request_producer(encoded_img)

    return {"message": "image sent successfully"}

def run_server():
    uvicorn.run(app, port=8000)

