import cv2
import numpy as np
import sqlite3
from PIL import Image
from datetime import datetime
import os

#Insert vao Database, lay du lieu nhan dien
def insertRecord(SoTu):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
    cusror = conn.execute(query)
    isRecordExist = 0
    for row in cusror:
        isRecordExist = 1
    if(isRecordExist == 0):
        query = "INSERT INTO data (SoTu,ThoiGian) VALUES (" + str(SoTu) + ",'" + str(datetime.now().strftime("%d.%m.%Y %H.%M")) + "')"
    else:
        query = "UPDATE data SET ThoiGian = '"+ str(datetime.now().strftime("%d.%m.%Y %H.%M")) +"' WHERE SoTu =" + str(SoTu)
    conn.execute(query)
    conn.commit()
    conn.close() 

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sampleNum = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if not os.path.exists("dataSet"):
                os.makedirs("dataSet")
            sampleNum += 1
            cv2.imwrite("dataSet/TuSo."+str(SoTu)+"."+str(sampleNum)+".jpg",gray[y : y+h,x : x+w])
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
        if sampleNum > 10:
            if not os.path.exists("dataFace"):
                os.makedirs("dataFace")
            cv2.imwrite("dataFace/TuSo."+str(SoTu) + "_" + str(datetime.now().strftime("%d.%m.%Y %H.%M")) + ".jpg",frame[y-80 : y+h+80,x-60 : x+w+60])
            break
    cap.release()
    cv2.destroyAllWindows()

#Xoa ban ghi trong database, xoa anh nhan dien
def deleteRecord(SoTu):
    conn = sqlite3.connect("database.db")
    query = "DELETE FROM data WHERE SoTu = " + str(SoTu)
    conn.execute(query)
    conn.commit()
    conn.close() 

    path = "dataSet"
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        if(SoTu == int(imagePath.split("\\")[1].split(".")[1])):
            os.remove(imagePath)

insertRecord(1)
#deleteRecord(1)