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
        return np.round(self.model.predict_proba(y_pred).max(), 2)
        # return np.round(np.max(self.model.predict(y_pred)), 2)
    
    def predict(self, img):
        y_pred = self.facenet.embeddings(img)
        return self.classes[int(self.model.predict(y_pred))].decode('utf-8')
    
class ImageData(BaseModel):
    image: str

