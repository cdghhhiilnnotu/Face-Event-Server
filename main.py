from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
import cv2
from PIL import Image
import io
import time
import asyncio
import nest_asyncio

from modules import *
from asmodules import *

app = FastAPI()

model_path = ".\\models\\all\\hau_face_model.pkl"
classes_path = ".\\models\\all\\hau_face_classes.npy"
embedded_path = ".\\models\\all\\hau_face_embedded.npz"

face_reg = FaceRecogition()
face_reg.load_models(model_path, embedded_path, classes_path)

@app.post("/predicting/")
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
    # img = [img, img]
    # # print(f"img: {img[0].shape}")
    # face_score = face_reg.predict_proba(img)
    # # face_score = str(face_score)
    # # print(f"face_score: {face_score}")
    # face_id = face_reg.predict(img)
    # # print(f"face_id: {face_id}")
    # output = [{"id": str(id_), "score": score} for id_, score in zip(face_id, face_score)]

    # output = asyncio.run(reg_main)
    # try:
    #     output = asyncio.run(reg_main)
    #     return JSONResponse(content=output, status_code=200)
    #     # return {"message": result}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    # try:
    #     loop = asyncio.get_event_loop()
    #     output = loop.run_until_complete(reg_main())
    # #     output = asyncio.run(reg_main)
    #     return JSONResponse(content=output, status_code=200)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    try:
        output = await reg_main(img)
        return JSONResponse(content=output, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # return JSONResponse(content={"id": "2055010153", "score": 0.99}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
