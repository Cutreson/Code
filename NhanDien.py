import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("recoginizer/trainningData.yml")
def getProfile(SoTu):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
    cusror = conn.execute(query)
    profile = None
    for row in cusror:
        profile = row
    conn.close()
    return profile
############################################################

def nhanDien():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        SoTu,confidence = recognizer.predict(roi_gray)
        if confidence < 30:
            profile = getProfile(SoTu)
            if(profile != None):
                print("True")
                cap.release()
                return True
        else:
            print("False")
            cap.release()
            return False

#####################################################
nhanDien()