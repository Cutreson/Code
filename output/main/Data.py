import cv2
import numpy as np
import sqlite3
from PIL import Image
from datetime import datetime
import time
import os

from Trainning import train_Data


def check_DataSet():
    pathss, dirs, files = next(os.walk("dataSet"))
    file_count = len(files)
    if(file_count == 0):
        return False
    else:
        return True

def check_Record():
    conn = sqlite3.connect("database.db")
    for SoTu in range(1,5):
        query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
        cusror = conn.execute(query)
        isRecordExist = 0
        for row in cusror:
            isRecordExist = 1
        if(isRecordExist == 0):
            conn.commit()
            conn.close() 
            print(SoTu)
            return SoTu
    conn.commit()
    conn.close() 
    print("0")
    return 0

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

def get_Face(SoTu):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sampleNum = 0
    time_out = time.time() + 10
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
            cap.release()
            cv2.destroyAllWindows()
            print("Lấy data thành công")
            return True
        if time.time() > time_out :
            cap.release()
            cv2.destroyAllWindows()
            print("Lấy data thất bại")
            return False
    

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

#########################
def getProfile(SoTu):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
    cusror = conn.execute(query)
    profile = None
    for row in cusror:
        profile = row
    conn.close()
    return profile
#####################################
def nhanDien():
    print("Nhan dien")
    if(check_DataSet() == False):
        return 0
    else:
        train_Data()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recoginizer/trainningData.yml")
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            SoTu,confidence = recognizer.predict(roi_gray)
            if confidence < 50:
                cap.release()
                return SoTu
            else:
                cap.release()
                return 0
######################################
#check_Record()
#insertRecord(4)
#get_Face(4)
deleteRecord(1)
deleteRecord(2)
deleteRecord(3)
deleteRecord(4)
#deleteRecord(2)
#check_DataSet()
print(nhanDien())