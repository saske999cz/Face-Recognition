from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream

import threading
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import argparse
import src.facenet as fn
import imutils
import os
import sys
import math
import pickle
import src.align.detect_face as df
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import time
import attendance

cred = credentials.Certificate("E:/Project/PBL4/serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL': "https://facerecognitionrealtime-default-rtdb.firebaseio.com/",
                               'storageBucket':"facerecognitionrealtime.appspot.com"}
                              )


        
        
      
def loading():
    win = tk.Tk()
    win.title("Loading")
    win.geometry("450x140+400+100")
    percent = StringVar()
    bar = Progressbar(win, orient = HORIZONTAL,length = 300)
    bar.pack(pady = 10)
    percentLabel = Label(win, textvariable = percent).pack()
    tasks = 500
    x = 0
    win.resizable(0,0)
    win.iconbitmap(r'E:/Project/PBL4/face2.ico')
    win.protocol("WM_DELETE_WINDOW", exit)
    while(x<tasks):
        time.sleep(0.02)
        bar['value']+=0.2
        x+=1
        percent.set(str(int((x/tasks)*100))+"%")
        win.update_idletasks()
        
    if(bar['value'] == 100):
        time.sleep(5)
        win.destroy()
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()
    imgBackground = cv2.imread("E:/Project/PBL4/App-ui/Img/background.png")
    folderModePath = 'E:/Project/PBL4/App-ui/Modes'
    modePathList = os.listdir(folderModePath)
    imgModeList = []
    imgAvatarList = []
    imgUnknown = cv2.imread("E:/Project/PBL4/App-ui/Img/unkown.png")
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    modeType = 0
    counter = 0
    bucket = storage.bucket()
    imgUser = []
    folderPath = 'E:/Project/PBL4/App-ui/Avatar'
    pathList = os.listdir(folderPath)
    #print(pathList)
    imgList = []
    studentIds = []
    
    for path in pathList:
        fileName = f'{folderPath}/{path}'
        #print(fileName)
        bucket1 = storage.bucket()
        blob1 = bucket1.blob(f'Avatar/{path}')
        blob1.upload_from_filename(fileName)
        
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'E:/Project/PBL4/Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'E:/Project/PBL4/Models/20180402-114759.pb'
   
   

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            fn.load_model(FACENET_MODEL_PATH)
            
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = df.create_mtcnn(sess, "E:/Project/PBL4/src/align")

            
            people_detected = set()
            
            person_detected = collections.Counter()
            
            
            cap  = VideoStream(src=0).start()
            
            prev_frame_time = 0
            new_frame_time = 0

            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=640)
                frame = cv2.flip(frame, 1)
                imgBackground[44:44+633, 808:808+414] = imgModeList[modeType]
                bounding_boxes, _ = df.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try: 
                    
                    if faces_found<=0:
                        modeType = 0
                        counter = 0  
                    elif faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                        modeType = 0
                        counter = 0
                        
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            # print(bb[i][3]-bb[i][1])
                            # print(frame.shape[0])
                            # print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = fn.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))



                                if best_class_probabilities > 0.80:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][1] - 10

                                    name = class_names[best_class_indices[0]]


                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x + 180, text_y),
                                                cv2.FONT_HERSHEY_COMPLEX,
                                                0.7, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    if counter == 0:
                                        cv2.putText(frame, "Loading%", (text_x + 50, text_y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 204, 0), thickness=1, lineType=2)
                                        cv2.waitKey(1)
                                        counter = 1
                                        modeType = 1
                                    
                                else:
                                    name = "Unknown"
                                    cv2.rectangle(
                                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(imgBackground,"?", (960, 494),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
                                    
                                    modeType = 1
                                    counter = 0
                                    
                                    
                                    

                except:
                    pass
                if counter != 0:
                    if counter == 1:
                        userInfo = db.reference(f"Users/{name}").get()
                        blob = bucket.get_blob(f"Avatar/{name}.png")
                        array = np.frombuffer(blob.download_as_string(), np.uint8)
                        imgUser = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                        datetimeObject = datetime.strptime(userInfo['last_checked'],
                                                   "%Y-%m-%d %H:%M:%S")
                        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                        #print(secondsElapsed)
                        if secondsElapsed > 30:
                            ref = db.reference(f'Users/{name}')
                            ref.child('last_checked').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        
                
                            
                    if modeType != 3:
 
                        if 25 < counter < 30:
                            modeType = 2
                            if counter == 26:
                                attendance.attendance(str(name))
        
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                        if counter <= 25:
                            cv2.putText(imgBackground,str(name), (960, 494),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
                            
                            imgBackground[175:175+216, 909:909+216] = imgUser
                        
                        counter += 1
                        if counter >= 30:
                            counter = 0
                            modeType = 0
                            userInfo = []
                            imgUser = []
                            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    
                
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(frame, "FPS: "+fps, (7, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 255, 0), 1, cv2.LINE_AA)
                
                imgBackground[162:162+480, 55:55+640] = frame
                
                
                cv2.imshow('Face Recognition', imgBackground)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            #cap.release()
            cv2.destroyAllWindows()


a = threading.Thread(target=loading, args=())
a.start()

if __name__ == "__main__":
    main()


