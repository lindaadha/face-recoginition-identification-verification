from cvzone.FaceDetectionModule import FaceDetector
from model.resnet import resnet_face18
from torch.nn import DataParallel
from sklearn.metrics.pairwise import cosine_similarity
import cv2, math, json, torch, os
import numpy as np

def load_model():
    model = resnet_face18(use_se=False)
    model = DataParallel(model)
    model_pretrained = torch.load("model/resnet18_110.pth", map_location="cpu")
    model.load_state_dict(model_pretrained)
    model.to(torch.device("cpu"))
    return model

def crop_face (image):
    img_ori = image.copy()
    img, bboxs = detector.findFaces(image)
    # print(bboxs[0]['bbox']) 
    x, y, w, h = bboxs[0]['bbox']
    crop = img_ori[y:y+h, x:x+w]
    crop = cv2.resize(crop,(120, 120))
    return crop 

def load_image(image):
    image = crop_face(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def FR_inference(processed_image):
    with torch.no_grad():
        data = torch.from_numpy(processed_image)
        data = data.to(torch.device("cpu"))
        output = model(data)
        output = output.data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1,fe_2))
    return feature

def calculate_angle(emb1, emb2):
    cos_sim = cosine_similarity(emb1, emb2)
    # cos_sim=cosine_similarity(emb1.reshape(1,-1),emb2.reshape(1,-1))
    angle = math.acos(cos_sim[0][0])
    angle = math.degrees(angle)
    return angle

def compare_two_faces(face1, face2):
    emb1 = FR_inference(face1)
    emb2 = FR_inference(face2)
    angle = calculate_angle(emb1, emb2)
    if angle < 60 :
        print("Image Match")
        id = recognition('result.json',emb2)
    else:
        print("Image doesn't Match")
        id = None
    return angle, id 

def recognition(dict, feature):
    with open(dict, "r") as outfile:
        dict_face =json.load(outfile)
    best = 360
    for key in dict_face:
        value = dict_face[key]
        value = np.array(value)
        angle = calculate_angle(feature, value)        
        if angle < best:
            best = angle
            best_id = key    
    # print(best)
    if best > 60:
        return 'unknown'
    return best_id


if __name__ == '__main__':
    model = load_model()
    model.eval()
    detector = FaceDetector()

    img1 = cv2.imread('test_face/mindy.jpg')
    img2 = cv2.imread('test_face/jerry.jpg')
    face1 = load_image(img1)
    face2 = load_image(img2)
    angle = compare_two_faces(face1, face2)
    print(angle)
