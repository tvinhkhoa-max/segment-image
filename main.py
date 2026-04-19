from fastapi import FastAPI, UploadFile, File
from rembg import remove
import base64
import numpy as np
import cv2
import base64
from sam_nail import extract_nail
from sam_auto_nail import extract_nail_auto

app = FastAPI()

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
  input_bytes = await file.read()
  output = remove(input_bytes)

  return {
    "image": base64.b64encode(output).decode()
  }

@app.post("/extract-nail")
async def extract(file: UploadFile = File(...)):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cv2.imwrite("tmp.png", img)

    result = extract_nail("tmp.png")
    result = extract_nail_auto("tmp.png")

    _, buffer = cv2.imencode(".png", result)

    return {
      "image": base64.b64encode(buffer).decode()
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
  input_bytes = await file.read()
  output = remove(input_bytes)

  npimg = np.frombuffer(output, np.uint8)
  img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

  # cv2.imwrite("tmp.png", img)

  # result = extract_nail("tmp.png")
  result = extract_nail_auto(img)

  _, buffer = cv2.imencode(".png", result)

  return {
    "image": base64.b64encode(buffer).decode()
  }