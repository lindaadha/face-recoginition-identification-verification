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
    angle = math.acos(cos_sim[0][0])
    angle = math.degrees(angle)
    return angle

if __name__ == "__main__":
    #load model
    model = load_model()
    model.eval()
    detector = FaceDetector()
    img = cv2.imread('guntur.jpg')
    img_crop = crop_face(img)
    img_crop = load_image(img_crop)
    feature = FR_inference(img_crop)

    print("Masukkan Nama:")
    nama = input()
    print (nama, feature)

    f = open('result2.json')
    data = json.load(f)
    f.close()

    data[nama] = feature.tolist()

    with open('result2.json',"w") as outfile:
        json.dump(data, outfile)
