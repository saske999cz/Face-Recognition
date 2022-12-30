from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream

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


import time


def findThreshold():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            fn.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = df.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            path = "TestData"
            myList = os.listdir(path)
            total_img = 0
            detection = 0
            recognition = 0
            threshold = 0
            for item in myList:
                frame = cv2.imread(f"{path}/{item}")
                total_img = total_img + 1
                frame = imutils.resize(frame, width=300)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = df.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            detection = detection + 1
                            # print(bb[i][3]-bb[i][1])
                            # print(frame.shape[0])
                            # print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:
                                cropped = frame[bb[i][1]:bb[i]
                                                [3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = fn.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(
                                    -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {
                                    images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(
                                    embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(
                                    predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(
                                    best_name, best_class_probabilities))
                                
                                threshold = threshold + best_class_probabilities

                                if best_class_probabilities > 0.8:
                                    recognition = recognition + 1
                                    cv2.rectangle(
                                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][1] - 10

                                    name = class_names[best_class_indices[0]]


                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x + 150, text_y),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                else:
                                    name = "Unknown"
                                    cv2.rectangle(
                                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX,
                                                1, (255, 255, 255), thickness=1, lineType=2)

                except:
                    pass

                # cv2.imshow('Face Recognition', frame)
                # cv2.waitKey()

            print("Total number of images:" + str(total_img))
            print("Number of images detected face: " + str(detection))
            print("Number of images correctly identified: " + str(recognition))
            print("Percent correct recognition: " + str(recognition/detection))
            # print("Threshold: " + str(threshold/detection))


findThreshold()
