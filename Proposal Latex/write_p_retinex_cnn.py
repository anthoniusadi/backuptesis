import numpy as np
from tensorflow.keras.models import model_from_json,load_model
from time import sleep
import operator
import cv2
import sys, os
from keras.utils.np_utils import to_categorical
from time import sleep
import retinex
import time 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dela/no retinex/ALL/ALL_Dela25lux.mp4',fourcc,25.0,(640,480))
json_file = open("mobile_4.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("mobile_4.h5")
cap = cv2.VideoCapture("/root/pengujian/video_pengujian/ASL_Dela25lux.mp4")
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}
labels_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
global mulai 
start= True
while start:
    try:
        _, fr = cap.read()
        frame = rescale_frame(fr,percent=100)
        x1 = int(0.5*frame.shape[1])+80
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])-80
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (224, 224)) 
        test_image = (roi)
        result = loaded_model.predict(test_image.reshape(1, 224, 224, 3)) 
        res = str(np.argmax(result))
        cv2.putText(frame,res ,(20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (20,10,255), 2)
        out.write(frame) 
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: 
            break
    except:
        selesai=time.time()
        print("waktu selesai",selesai)
        print("done")
        waktukomp= selesai - mulai
        print(" waktu komputasi ",waktukomp)
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        start=False
# cap.release()
# cv2.destroyAllWindows()
# out.release()
