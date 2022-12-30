from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import src.facenet as fn
import os
import math
import pickle
from sklearn.svm import SVC

use_split_dataset = False
data_dir = 'E:/Project/PBL4/Dataset/FaceData/processed'
classifier_filename = 'E:/Project/PBL4/Models/facemodel.pkl'
model = 'E:/Project/PBL4/Models/20180402-114759.pb'

def trainModel(mode='TRAIN'):

    with tf.Graph().as_default():

        with tf.compat.v1.Session() as sess:

            np.random.seed(seed=666)

            if use_split_dataset:
                dataset_tmp = fn.get_dataset(data_dir)
                train_set, test_set = split_dataset(dataset_tmp, 20, 10)
                if (mode == 'TRAIN'):
                    dataset = train_set
                elif (mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = fn.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths) > 0,
                       'There must be at least one image for each class in the dataset')

            paths, labels = fn.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            fn.load_model('E:/Project/PBL4/Models/20180402-114759.pb')

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / 1000))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*1000
                end_index = min((i+1)*1000, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = fn.load_data(paths_batch, False, False, 160)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            if (mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' %
                      classifier_filename_exp)

            elif (mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' %
                      classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (
                        i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(fn.ImageClass(
                cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(fn.ImageClass(
                cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set



trainModel(mode='TRAIN')
