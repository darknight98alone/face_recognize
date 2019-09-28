from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, argparse ,time ,sys, configparser
from os.path import basename
import gc
import cv2, sklearn
from utils import face_preprocess
from nets.mtcnn_model import P_Net, R_Net, O_Net
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector

from multiprocessing import Process, Queue
from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
import threading

def load_mobilefacenet(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
def load_model(model="./model/weight.pb"):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1- feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

def readImage(path):
    nimg = cv2.imread(path)
    nimg = cv2.resize(nimg,(160,160))
    nimg = nimg - 127.5
    nimg = nimg * 0.0078125
    return nimg

def load_faces(faces_dir, mtcnn_detector):
    face_db = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # load_mobilefacenet("./models/mobilefacenet_model/MobileFaceNet_9925_9680.pb")
            load_model()
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for root, dirs, files in os.walk(faces_dir):
                for file in files:
                    # nimg = cv2.imread(os.path.join(root, file))
                    nimg = np.zeros((1, 160, 160, 3))
                    # nimg = nimg - 127.5
                    # nimg = nimg * 0.0078125
                    name = basename(root)
                    input_image = np.expand_dims(nimg,axis=0)
                    print(input_image.shape)
                    feed_dict = {inputs_placeholder: input_image,phase_train_placeholder:False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    embedding = sklearn.preprocessing.normalize(emb_array).flatten()
                    face_db.append({
                        "name": name,
                        "feature": embedding
                    })
    return face_db

def main(a,b):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_mobilefacenet("./model/20180402-114759.pb")
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            input_images = np.zeros((2, 160,160,3))
            input_images[0] = readImage(a) 
            input_images[1] = readImage(b)
            feed_dict = {inputs_placeholder: input_images,phase_train_placeholder:False}
            emb_arrays = sess.run(embeddings, feed_dict=feed_dict)
            emb_arrays = sklearn.preprocessing.normalize(emb_arrays)   
            embedding1 = emb_arrays[0].flatten()
            embedding2 = emb_arrays[1].flatten()
            ret, sim = feature_compare(embedding1,embedding2, 0.65) 
            print(sim)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--imageDir1",help="path to image 1")
    parser.add_argument("-i2", "--imageDir2",help="path to image 2")
    args = parser.parse_args()
    main(args.imageDir1,args.imageDir2)