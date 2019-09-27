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
import time

def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1- feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

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
def load_faces(faces_dir, mtcnn_detector):
    face_db = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_mobilefacenet("./models/mobilefacenet_model/MobileFaceNet_9925_9680.pb")
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            for root, dirs, files in os.walk(faces_dir):
                index=1
                # for file in files:
                for file in sorted(files):
                    nimg = cv2.imread(os.path.join(root, file))
                    nimg = nimg - 127.5
                    nimg = nimg * 0.0078125
                    name = basename(root)
                    input_image = np.expand_dims(nimg,axis=0)
                    feed_dict = {inputs_placeholder: input_image}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    embedding = sklearn.preprocessing.normalize(emb_array).flatten()
                    face_db.append({
                        "name": name + "_" + str(index),
                        # "name": name, 
                        "feature": embedding
                    })
                    index = index + 1
    return face_db

def similarIOU(box1,box2):
    (x1,y1,x2,y2)= box1
    (a1,b1,a2,b2) = box2
    tx = (x1+x2)/2
    ty = (y1+y2)/2
    ta = (a1+a2)/2
    tb = (b1+b2)/2  
    temp = 20
    if (a1 - temp<=tx<=a2+temp and b1-temp<= ty<=b2+temp) or (x1-temp<=ta<=x2+temp and y1-temp<=tb<=y2+temp):
        return True
    print("iou err")
    return False

class trackedFace():
    """ Class for store face image embbedding and name for a face detected
    """
    def __init__(self,name):
        self.name = name
        self.countDisappeared = 0
        self.listImage = []
        self.listEmbedding = []
        self.justAdded = False
        self.latestBox = []
        self.liveTime = 0
        self.latestFrameCounter = 0

    def saveTrackedFace(self,savePath,startId):
        incStartId = True
        if not os.path.isdir(savePath):
            print("save fail")
        else:
            if RepresentsInt(self.name):
                startId = "person_"+ str(startId)
            else:
                startId = self.name
            if not os.path.isdir(os.path.join(savePath,startId)):
                os.mkdir(os.path.join(savePath,startId))
            else:
                incStartId = False
            numberImage = len(next(os.walk(os.path.join(savePath,startId)))[2])
            if numberImage <= 20:
                i = 0
                for index,image in enumerate(self.listImage):
                    if numberImage + index + 1 >20:
                        break
                    while True:
                        if not os.path.exists(os.path.join(savePath,startId+"/"+str(i)+".jpg")):
                            cv2.imwrite(os.path.join(savePath,startId+"/"+str(i)+".jpg"),image)
                            break
                        else:
                            i = i + 1
                print("save success: " + os.path.join(savePath,startId))
            else:
                print(os.path.join(savePath,startId) + " had enough data")
        return incStartId
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
def measureGoodFace(MIN_FACE_SIZE,MAX_BLACK_PIXEL,image,yaw,pitch,roll,BLUR_THRESH,YAWL,YAWR,PITCHL,PITCHR,ROLLL,ROLLR):
    notBlur= cv2.Laplacian(image, cv2.CV_64F).var()
    # print(notBlur)
    if image.shape[0]<MIN_FACE_SIZE or image.shape[1]<MIN_FACE_SIZE:
        # print("small face")
        return False
    if notBlur < BLUR_THRESH:
        # print("blur")
        return False
    if yaw<YAWL or yaw>YAWR or pitch<PITCHL or pitch>PITCHR or roll<ROLLL or roll>ROLLR:
        print("headpose err")
        return False
    image = image.astype('float32')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.shape[0]*image.shape[1]-cv2.countNonZero(image) > MAX_BLACK_PIXEL:
        # print("black")
        return False
    # print("good face")
    return True 

def countIdFolder(path):
    count = 0
    if os.path.isdir(path):
        for name in next(os.walk(path))[1]:
            if (len(name)>7):
                name = name[7:]
                if RepresentsInt(name):
                    count = count + 1
    return count
import shutil
def main(strargument):
    shutil.rmtree("./test")
    os.mkdir("./test")
    os.remove("result.txt")
    f = open("result.txt", "a")
    cap = cv2.VideoCapture(strargument)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap = cv2.VideoCapture("NTQ.mkv")
    #cap = cv2.VideoCapture("/home/fitmta/Real-Time-Face-Detection-OpenCV-GPU/videos/video/out1.1.avi")
    #cap = cv2.VideoCapture("http://root:1234Qwer!@#$@27.72.105.10:8932/mjpg/video.mjpg")
    # cap = cv2.VideoCapture("http://operator:Abc@12345@27.72.105.10:8933/Streaming/channels/102/preview")
    success, frame = cap.read()
    startId = countIdFolder("./face_db/")
    # quit if unable to read the video file
    if not success:
      print('Failed to read video')
      sys.exit(1)
    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)
    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0
    #Variables holding the correlation trackers and the name per faceid
    conf = configparser.ConfigParser()
    conf.read("config/main.cfg")
    mtcnn_detector = load_mtcnn(conf)
    MODEL_PATH = conf.get("MOBILEFACENET", "MODEL_PATH")
    VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
    FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")
    BLUR_THRESH = int(conf.get("CUSTOM","BLUR_THRESH"))
    MIN_FACE_SIZE= int(conf.get("MTCNN","MIN_FACE_SIZE"))
    MAX_BLACK_PIXEL = int(conf.get("CUSTOM","MAX_BLACK_PIXEL"))
    YAWL= int(conf.get("CUSTOM","YAWL"))
    YAWR= int(conf.get("CUSTOM","YAWR"))
    PITCHL= int(conf.get("CUSTOM","PITCHL"))
    PITCHR= int(conf.get("CUSTOM","PITCHR"))
    ROLLL= int(conf.get("CUSTOM","ROLLL"))
    ROLLR= int(conf.get("CUSTOM","ROLLR"))
    MAXDISAPPEARED=int(conf.get("CUSTOM","MAXDISAPPEARED"))
    IS_FACE_THRESH=float(conf.get("CUSTOM","IS_FACE_THRESH"))
    EXTEND_Y =int(conf.get("CUSTOM","EXTEND_Y"))
    EXTEND_X=int(conf.get("CUSTOM","EXTEND_X"))
    SIMILAR_THRESH=float(conf.get("CUSTOM","SIMILAR_THRESH"))
    MAX_LIST_LEN =int(conf.get("CUSTOM","MAX_LIST_LEN"))
    MIN_FACE_FOR_SAVE=int(conf.get("CUSTOM","MIN_FACE_FOR_SAVE"))
    LIVE_TIME=int(conf.get("CUSTOM","LIVE_TIME"))
    ROIXL= int(conf.get("CUSTOM","ROIXL"))
    ROIXR= int(conf.get("CUSTOM","ROIXR"))
    ROIYB= int(conf.get("CUSTOM","ROIYB"))
    ROIYA= int(conf.get("CUSTOM","ROIYA"))
    maxDisappeared =MAXDISAPPEARED  ## khong xuat hien toi da 100 frame
    faces_db = load_faces(FACE_DB_PATH, mtcnn_detector)
    # load_face_db = ThreadingUpdatefacedb(FACE_DB_PATH,mtcnn_detector)
    time.sleep(10)
    for item in faces_db:
        print(item["name"])
    listTrackedFace = []
    mark_detector = MarkDetector()
    tm = cv2.TickMeter()
    _, sample_frame = cap.read()
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_mobilefacenet(MODEL_PATH)
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            try:
                start = time.time()
                while True:
                    start1 = time.time()
                    retval, frame = cap.read()
                    
                    #Increase the framecounter
                    frameCounter += 1
                    if retval:
                        _frame = frame[ROIYA:ROIYB,ROIXL:ROIXR]
                        cv2.rectangle(frame, (ROIXL, ROIYA), (ROIXR, ROIYB), (0, 0, 255), 2)
                        good_face_index = []
                        # faces_db = load_face_db.face_db
                        if (frameCounter % 1) == 0:
                            ### embed and compare name 
                            for i,face_db in enumerate(faces_db):
                                if not os.path.isdir("./face_db/"+face_db["name"].split("_")[0]):
                                    faces_db.pop(i)
                            faces,landmarks = mtcnn_detector.detect(_frame)
                            if faces.shape[0] is not 0:
                                input_images = np.zeros((faces.shape[0], 112,112,3))
                                save_images = np.zeros((faces.shape[0], 112,112,3))
                                (yaw,pitch, roll) = (0,0,0)
                                for i, face in enumerate(faces):
                                    if round(faces[i, 4], 6) > IS_FACE_THRESH:
                                        bbox = faces[i,0:4]
                                        points = landmarks[i,:].reshape((5,2))
                                        nimg = face_preprocess.preprocess(_frame, bbox, points, image_size='112,112')
                                        save_images[i,:] = nimg
                                        nimg = nimg - 127.5
                                        nimg = nimg * 0.0078125
                                        input_images[i,:] = nimg
                                        (x1,y1,x2,y2) = bbox.astype("int")
                                        if x1<0 or y1<0 or x2<0 or y2<0 or x1 >=x2 or y1 >= y2:
                                            continue
                                        temp = int((y2-y1)/EXTEND_Y)
                                        y1 = y1 + temp
                                        y2 = y2 + temp
                                        temp = int((x2-x1)/EXTEND_X)
                                        if x1>temp:
                                            x1 = x1 - temp
                                        x2 = x2 + temp
                                        # cv2.imshow("mainframe",frame)
                                        # cv2.imwrite("temp2.jpg",frame[y1:y2,x1:x2])
                                        face_img = cv2.resize(_frame[y1:y2,x1:x2], (128, 128))
                                        # cv2.imshow("ok",face_img)
                                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                        tm.start()
                                        marks = mark_detector.detect_marks([face_img])
                                        tm.stop()
                                        marks *= (x2 -x1)
                                        marks[:, 0] += x1
                                        marks[:, 1] += y1
                                        # mark_detector.draw_marks(
                                        #         frame, marks, color=(0, 255, 0))
                                        pose, (yaw,pitch,roll) = pose_estimator.solve_pose_by_68_points(marks)
                                        # temp = frame
                                        # cv2.putText(temp,"yaw:  "+str(yaw),(x2,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)
                                        # cv2.putText(temp,"pitch: "+str(pitch),(x2,y1+25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)
                                        # cv2.putText(temp,"roll:   "+str(roll),(x2,y1+50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)
                                        # cv2.imshow("frame",temp)
                                        # if measureGoodFace(MIN_FACE_SIZE,MAX_BLACK_PIXEL,frame[y1:y2,x1:x2],yaw,pitch,roll,BLUR_THRESH,YAWL,YAWR,PITCHL,PITCHR,ROLLL,ROLLR):
                                        #     good_face_index.append(i)
                                        # cv2.waitKey(0)
                                # print(good_face_index)
                                feed_dict = {inputs_placeholder: input_images}
                                emb_arrays = sess.run(embeddings, feed_dict=feed_dict)
                                emb_arrays = sklearn.preprocessing.normalize(emb_arrays)
                                names = []
                                sims = []
                                for i, embedding in enumerate(emb_arrays):
                                    # if len(listTrackedFace)>i and RepresentsInt(listTrackedFace[i].name)==False:
                                    #     names.append(listTrackedFace[i].name)
                                    #     continue
                                    embedding = embedding.flatten()
                                    temp_dict = {}
                                    for com_face in faces_db:
                                        ret, sim = feature_compare(embedding, com_face["feature"], 0.65)
                                        temp_dict[com_face["name"]] = sim
                                    # print(temp_dict)
                                    dictResult = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
                                    # print(dictResult[:5])
                                    name = ""
                                    if len(dictResult)>0 and dictResult[0][1] > VERIFICATION_THRESHOLD:
                                        name = dictResult[0][0]#.split("_")[0]
                                        sim = dictResult[0][1]
                                        ## wite log
                                        t = time.time()
                                        f.write(name+"___"+str((t - start)//60)+":"+str(int(t- start)%60)+"\n")
                                    else:
                                        name = "unknown"
                                        sim = 0
                                    names.append(name)
                                    sims.append(sim)
                                    
                                    # cv2.imwrite("./test/"+name+"_"+str(frameCounter//60)+":"+str(frameCounter%60)+".jpg",save_images[i,:])
                                    # if len(dictResult)>0 :
                                        # cv2.imwrite("./test/"+names[i]+"_"+str(frameCounter//60)+":"+str(frameCounter%60)+"_"+str(dictResult[0][1])+".jpg",save_images[i,:])
                                    ################################ tracker
                                for i, embedding in enumerate(emb_arrays):
                                    embedding = embedding.flatten()
                                    ResultDict = {}
                                    for objectTrackFace in listTrackedFace:
                                        tempList = []
                                        (x1,y1,x2,y2) = objectTrackFace.latestBox
                                        for com_face in objectTrackFace.listEmbedding:
                                            ret, sim = feature_compare(embedding, com_face, 0.65)
                                            tempList.append(sim)
                                        tempList.sort(reverse=True)
                                        if len(tempList)>0:
                                            if tempList[0]> 0.9 or (similarIOU(faces[i,:4].astype("int"),objectTrackFace.latestBox) and (frameCounter - objectTrackFace.latestFrameCounter)<3):
                                                ResultDict[objectTrackFace.name] = tempList[0]
                                    dictResult = sorted(ResultDict.items(), key=lambda d: d[1], reverse=True)
                                    if True:
                                        if len(ResultDict)>0 and dictResult[0][1] > SIMILAR_THRESH: ## neu khop -- 0.5
                                            # for ik in range(len(dict)):
                                            #     if dict[ik][1]>SIMILAR_THRESH:
                                            
                                            nameTrackCurrent = dictResult[0][0]
                                            for index,tempFaceTrack in enumerate(listTrackedFace):
                                                if tempFaceTrack.name == nameTrackCurrent:
                                                    if len(tempFaceTrack.listImage)>MAX_LIST_LEN:
                                                        tempFaceTrack.listImage.pop(0)
                                                        tempFaceTrack.listEmbedding.pop(0)
                                                        if measureGoodFace(MIN_FACE_SIZE,MAX_BLACK_PIXEL,save_images[i,:],yaw,pitch,roll,BLUR_THRESH,YAWL,YAWR,PITCHL,PITCHR,ROLLL,ROLLR):
                                                            tempFaceTrack.listImage.append(save_images[i,:])
                                                            tempFaceTrack.listEmbedding.append(emb_arrays[i])
                                                    else:
                                                        if measureGoodFace(MIN_FACE_SIZE,MAX_BLACK_PIXEL,save_images[i,:],yaw,pitch,roll,BLUR_THRESH,YAWL,YAWR,PITCHL,PITCHR,ROLLL,ROLLR):
                                                            tempFaceTrack.listImage.append(save_images[i,:])
                                                            tempFaceTrack.listEmbedding.append(emb_arrays[i])
                                                    if names[i] != "unknown":
                                                        if RepresentsInt(nameTrackCurrent):
                                                            tempFaceTrack.name = names[i]
                                                        # else: #################
                                                        #     names[i] = nameTrackCurrent
                                                    else:
                                                        if not RepresentsInt(nameTrackCurrent):
                                                            names[i] = nameTrackCurrent
                                                    tempFaceTrack.countDisappeared = 0
                                                    tempFaceTrack.latestBox = faces[i,0:4].astype("int")
                                                    tempFaceTrack.latestFrameCounter = frameCounter
                                                    tempFaceTrack.liveTime = 0
                                                    tempFaceTrack.justAdded = True ## but we still action with it
                                                    break
                                                    
                                        else: ## neu khong khop thi tao moi nhung chi them anh khi mat du tot
                                            if len(ResultDict)>0:
                                                print(dictResult[0][1])
                                            if names[i] != "unknown":
                                                newTrackFace = trackedFace(names[i])
                                            else:
                                                newTrackFace = trackedFace(str(currentFaceID))
                                                currentFaceID = currentFaceID + 1
                                            if measureGoodFace(MIN_FACE_SIZE,MAX_BLACK_PIXEL,save_images[i,:],yaw,pitch,roll,BLUR_THRESH,YAWL,YAWR,PITCHL,PITCHR,ROLLL,ROLLR):
                                                newTrackFace.listImage.append(save_images[i,:])
                                                newTrackFace.listEmbedding.append(emb_arrays[i])
                                            newTrackFace.latestBox = faces[i,0:4].astype("int")
                                            newTrackFace.latestFrameCounter = frameCounter
                                            # print(newTrackFace.latestBox)
                                            newTrackFace.justAdded = True
                                            listTrackedFace.append(newTrackFace) ## add list
                                ### disappeared
                                for index,trackFace in enumerate(listTrackedFace):
                                    if trackFace.justAdded == False:
                                        trackFace.countDisappeared = trackFace.countDisappeared + 1
                                        trackFace.liveTime = trackFace.liveTime + 1
                                    else:
                                        trackFace.justAdded = False
                                    if trackFace.liveTime > LIVE_TIME:
                                        t = listTrackedFace.pop(index)
                                        del t
                                    if trackFace.countDisappeared > maxDisappeared:
                                        if len(trackFace.listImage)<MIN_FACE_FOR_SAVE: ## neu chua duoc it nhat 5 mat thi xoa luon
                                            trackedFace.countDisappeared = 0 
                                            continue
                                        if trackFace.saveTrackedFace("./temp/",startId):
                                            startId = startId + 1
                                        t = listTrackedFace.pop(index)
                                        del t
                                for i,face in enumerate(faces):
                                    x1, y1, x2, y2 = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                                    x1 = max(int(x1), 0)
                                    y1 = max(int(y1), 0)
                                    x2 = min(int(x2), _frame.shape[1])
                                    y2 = min(int(y2), _frame.shape[0])
                                    cv2.rectangle(frame, (x1+ROIXL, y1+ROIYA), (x2+ROIXL, y2+ROIYA), (0, 255, 0), 2)
                                    # if i in good_face_index:
                                    # if not RepresentsInt(names[i]):
                                    cv2.putText(frame, names[i].split("_")[0] , (int(x1/2 + x2/2+ROIXL), int(y1+ROIYA)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
                            else:
                                for index,trackFace in enumerate(listTrackedFace):
                                    trackFace.countDisappeared = trackFace.countDisappeared + 1
                                    trackFace.liveTime = trackFace.liveTime + 1
                                    if trackFace.liveTime > LIVE_TIME:
                                        t = listTrackedFace.pop(index)
                                        del t
                                        continue
                                    if trackFace.countDisappeared > maxDisappeared:
                                        if len(trackFace.listImage)<MIN_FACE_FOR_SAVE: ## neu chua duoc it nhat 5 mat thi xoa luon
                                            trackedFace.countDisappeared = 0 
                                            continue
                                        if trackFace.saveTrackedFace("./temp/",startId):
                                            startId = startId + 1
                                        t = listTrackedFace.pop(index)
                                        del t
                            end = time.time()
                            cv2.putText(frame, "FPS: "+ str(1//(end-start1)), (400,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 3)
                            cv2.putText(frame, "Time: "+str((end - start)//60)+":"+str(int(end- start)%60), (200,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,0), 3)
                        cv2.imshow("frame", frame)
                        key = cv2.waitKey(30)
                        if key & 0xFF == ord('q'):
                                break
                        if key == 32:
                                cv2.waitKey(0)
                    else:
                        break
            except KeyboardInterrupt as e:
                pass
    gc.collect()
    cv2.destroyAllWindows()
    exit(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video",help="path to video")
    parser.add_argument("-c", "--camera",help="camera devices")
    args = parser.parse_args()
    # if os.path.exists(args["video"]):
    #     main(args["video"],1)
    # else:
    #     main(args["camera"],0)
    if args.video  :
        main(args.video)
    elif args.camera:
        main(int(args.camera))
    else:
        print("wrong argument")
