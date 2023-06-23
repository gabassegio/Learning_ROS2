import cv2
import time
import json
from u2net import mask
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageFile
import glob
import os



BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def pose_points(path):
    image_width=600
    image_height=600

    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    threshold=0.2

    img = cv2.imread(path)
    photo_height=img.shape[0]
    photo_width=img.shape[1]
    net.setInput(cv2.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    out = net.forward()
    out = out[:, :19, :, :] 

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)


    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()

    cv2.imwrite('results/keypoints.png',img)
    with open('points.json', 'w') as f:
        json.dump(points, f)
    

def body_points(path):

    null = 'null'
    with open('points.json', 'r') as f:
        point = json.load(f)

    ##Neck
    if point[1]!=null:
        neck = point[1]
    
    ##Shoulder
    if point[2]!= null:
        Rshoulder = point[2]
    if point[5] != null:
        Lshoulder = point[5]

    ##Hips(waist)
    try:
        if point[8]!=null:
            waist = point[8][1]
    except:   
        try:
            if point[11]!=null:
                waist = point[11][1]
        except:
            waist = neck[1]-((Rshoulder[0]-neck[0])*3)
            print("exception")

    ##Knees
    try:
        if point[9]!=null:
            knee = point[9][1]
    except:   
        try:
            if point[12]!=null:
                knee = point[12][1]
        except:
            knee = waist-((Rshoulder[0]-neck[0])*2)
            print("exception")
    

    img = cv2.imread(path)
    img = img[neck[1]-100:knee,Rshoulder[0]-30:Lshoulder[0]+30]
    cv2.imwrite('results/imCrop.png',img)

    
    way = glob.glob('images/*')
    for py_file in way:
        try:
            os.remove(py_file)
        except OSError as e:
            rospy.logerr(f"Error:{ e.strerror}")

    modelo = Image.open('results/imCrop.png')
    width, height = modelo.size


    for i in range(3):
        if i==0:
            bodypart = modelo.crop((0, neck[1], width, waist))
            print(bodypart)
            bodypart.save('images/torso.png',bodypart)
        elif i ==1:
            bodypart = img[:0,waist:]
            cv2.imwrite('images/legs.png',bodypart)
        else:
            bodypart = img[:0,neck[1]:]
            cv2.imwrite('images/head.png',bodypart)

def creatingMask(path):
    output = mask()
    output=load_img(output)
    RESCALE = 255
    out_img = img_to_array(output) / RESCALE
    out_img = cv2.resize(out_img, (640, 480))

    THRESHOLD = 0.2
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    shape = out_img.shape
    a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
    mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
    a_layer = mul_layer * a_layer_init
    rgba_out = np.append(out_img, a_layer, axis=2)
    

    original_image = load_img(path)
    inp_img = img_to_array(original_image)
    
    inp_img = inp_img /RESCALE

    a_layer = np.ones(shape=(shape[0], shape[1], 1))

    rgba_inp = np.append(inp_img, a_layer, axis=2)

    # simply multiply the 2 rgba images to remove the backgound
    rem_back = (rgba_inp * rgba_out)
    rem_back_scaled = Image.fromarray((rem_back * RESCALE).astype('uint8'), 'RGBA')
    # save the resulting image to colab
    rem_back_scaled.save('results/removed_background.png')




if __name__ == '__main__':
    time.sleep(2)
    creatingMask('base/img.jpg')

    pose_points('results/removed_background.png')
    body_points('results/removed_background.png')


