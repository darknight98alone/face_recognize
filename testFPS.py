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
import time

def load_mtcnn(conf):
    # load mtcnn model
    MODEL_PATH = conf.get("MTCNN", "MODEL_PATH")
    MIN_FACE_SIZE = int(conf.get("MTCNN", "MIN_FACE_SIZE"))
    STEPS_THRESHOLD = [float(i)  for i in conf.get("MTCNN", "STEPS_THRESHOLD").split(",")]

    detectors = [None, None, None]
    prefix = [MODEL_PATH + "/PNet_landmark/PNet",
              MODEL_PATH + "/RNet_landmark/RNet",
              MODEL_PATH + "/ONet_landmark/ONet"]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=MIN_FACE_SIZE, threshold=STEPS_THRESHOLD)
    return mtcnn_detector
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

def main():
    cap = cv2.VideoCapture(0)
    conf = configparser.ConfigParser()
    conf.read("config/main.cfg")
    mtcnn_detector = load_mtcnn(conf)
    MODEL_PATH = conf.get("MOBILEFACENET", "MODEL_PATH")
    VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
    FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # load_mobilefacenet(MODEL_PATH)
            # inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            try:
                while True:
                    start = time.time()
                    retval, frame = cap.read()
                    #Increase the framecounter
                    if retval:
                        faces,landmarks = mtcnn_detector.detect(frame)
                        if faces.shape[0] is not 0:
                            for i, face in enumerate(faces):
                                if round(faces[i, 4], 6) > 0.96:
                                    x1, y1, x2, y2 = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                                    x1 = max(int(x1), 0)
                                    y1 = max(int(y1), 0)
                                    x2 = min(int(x2), frame.shape[1])
                                    y2 = min(int(y2), frame.shape[0])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    end = time.time()
                    cv2.putText(frame,"FPS"+str(1/(start-end))  , (10, 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                            break
                    if key == 32:
                            cv2.waitKey(0)
            except KeyboardInterrupt as e:
                pass
    gc.collect()
    cv2.destroyAllWindows()
    exit(0)
if __name__ == '__main__':
    main()