import cv2
import numpy as np
import os
from PIL import Image

path = "dataSet"
def train_Data():
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    pathss, dirs, files = next(os.walk("dataSet"))
    file_count = len(files)
    if(file_count == 0):
        return False
    else:
        faces = []
        SoTus = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert("L")
            faceNp = np.array(faceImg,"uint8")
            #print(faceNp)
            SoTu = int(imagePath.split("\\")[1].split(".")[1])
            faces.append(faceNp)
            SoTus.append(SoTu)
            cv2.imshow("trainning",faceNp)
            cv2.waitKey(5)
        cv2.destroyAllWindows()

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(SoTus))
        if not os.path.exists("recoginizer"):
            os.makedirs("recoginizer")
        recognizer.save("recoginizer/trainningData.yml")
        return True
print(train_Data())