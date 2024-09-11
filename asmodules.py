import asyncio
from modules import *

model_s20 = ".\\models\\v2\\s20\\hau_face_model.pkl"
classes_s20 = ".\\models\\v2\\s20\\hau_face_classes.npy"
embedded_s20 = ".\\models\\v2\\s20\\hau_face_embedded.npz"

face_s20 = FaceRecogition()
face_s20.load_models(model_s20, embedded_s20, classes_s20)
# -------------------------------------------------------------

model_s21 = ".\\models\\v2\\s21\\hau_face_model.pkl"
classes_s21 = ".\\models\\v2\\s21\\hau_face_classes.npy"
embedded_s21 = ".\\models\\v2\\s21\\hau_face_embedded.npz"

face_s21 = FaceRecogition()
face_s21.load_models(model_s21, embedded_s21, classes_s21)
# -------------------------------------------------------------

model_s22 = ".\\models\\v2\\s22\\hau_face_model.pkl"
classes_s22 = ".\\models\\v2\\s22\\hau_face_classes.npy"
embedded_s22 = ".\\models\\v2\\s22\\hau_face_embedded.npz"

face_s22 = FaceRecogition()
face_s22.load_models(model_s22, embedded_s22, classes_s22)
# -------------------------------------------------------------

model_s23 = ".\\models\\v2\\s23\\hau_face_model.pkl"
classes_s23 = ".\\models\\v2\\s23\\hau_face_classes.npy"
embedded_s23 = ".\\models\\v2\\s23\\hau_face_embedded.npz"

face_s23 = FaceRecogition()
face_s23.load_models(model_s23, embedded_s23, classes_s23)
# -------------------------------------------------------------

model_sgv = ".\\models\\v2\\sgv\\hau_face_model.pkl"
classes_sgv = ".\\models\\v2\\sgv\\hau_face_classes.npy"
embedded_sgv = ".\\models\\v2\\sgv\\hau_face_embedded.npz"

face_sgv = FaceRecogition()
face_sgv.load_models(model_sgv, embedded_sgv, classes_sgv)
# -------------------------------------------------------------


async def s20_reg(img):
    s20_score = face_s20.predict_proba(img)
    s20_id = face_s20.predict(img)
    return [{"id": str(id_), "score": score} for id_, score in zip(s20_id, s20_score)]

async def s21_reg(img):
    s21_score = face_s21.predict_proba(img)
    s21_id = face_s21.predict(img)
    return [{"id": str(id_), "score": score} for id_, score in zip(s21_id, s21_score)]

async def s22_reg(img):
    s22_score = face_s22.predict_proba(img)
    s22_id = face_s22.predict(img)
    return [{"id": str(id_), "score": score} for id_, score in zip(s22_id, s22_score)]

async def s23_reg(img):
    s23_score = face_s23.predict_proba(img)
    s23_id = face_s23.predict(img)
    return [{"id": str(id_), "score": score} for id_, score in zip(s23_id, s23_score)]

async def sgv_reg(img):
    sgv_score = face_sgv.predict_proba(img)
    sgv_id = face_sgv.predict(img)
    return [{"id": str(id_), "score": score} for id_, score in zip(sgv_id, sgv_score)]

async def reg_main(img):
    results = await asyncio.gather(
        s20_reg(img),
        s21_reg(img),
        s22_reg(img),
        s23_reg(img),
        sgv_reg(img)
    )
    

    all_res = list(zip(*results))
    print(all_res)

    max_pred = [max(items, key=lambda x: x['score']) for items in all_res]

    return max_pred

