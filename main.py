from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import base64
import cv2
from PIL import Image
import io
import time

from modules import *

app = FastAPI()

model_path = ".\\models\\hau_face_model.pkl"
classes_path = ".\\models\\hau_face_classes.npy"
embedded_path = ".\\models\\hau_face_embedded.npz"

face_reg = FaceRecogition()
face_reg.load_models(model_path, embedded_path, classes_path)

@app.get("/predicting/")
async def upload_image(data: ImageData):
# async def upload_image():
    face_score = "-1"
    face_id = "undefined"

    img_data = base64.b64decode(data.image)
    img = Image.open(io.BytesIO(img_data))
    # img = Image.open("test1.png")
    img = np.array(img)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    # print(f"img: {img[0].shape}")
    face_score = face_reg.predict_proba(img)
    face_score = str(face_score)
    # print(f"face_score: {face_score}")
    face_id = face_reg.predict(img)
    # print(f"face_id: {face_id}")

    # return JSONResponse(content={"id": "2055010153", "score": 0.99}, status_code=200)
    return JSONResponse(content={"id": face_id, "score": face_score}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
