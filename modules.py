from keras_facenet import FaceNet
import pickle
import numpy as np
from pydantic import BaseModel

class FaceRecogition:

    def __init__(self):
        self.model = None
        self.class_embed = None
        self.classes = None
        self.facenet = FaceNet()

    def load_models(self, model_path, class_embed_path, classes_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.class_embed = np.load(class_embed_path)
        self.classes = np.load(classes_path)

    def predict_proba(self, img):
        y_pred = self.facenet.embeddings(img)
        id = [str(np.round(id_pred.max())) for id_pred in self.model.predict_proba(y_pred)]
        return id
        # return np.round(np.max(self.model.predict(y_pred)), 2)
    
    def predict(self, img):
        y_pred = self.facenet.embeddings(img)
        classes = [self.classes[int(pred)].decode('utf-8') for pred in list(self.model.predict(y_pred))]
        # self.classes = self.classes[int(self.model.predict(y_pred))]
        return classes
    
class ImageData(BaseModel):
    image: str

