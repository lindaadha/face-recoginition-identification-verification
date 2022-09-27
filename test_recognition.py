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

def euclidean (fit1, fit2):
    dist = np.linalg.norm(fit1 - fit2)
    return dist

def compare_two_faces(face1, face2):
    emb1 = FR_inference(face1)
    emb2 = FR_inference(face2)
    angle = euclidean(emb1, emb2)
    return angle

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


if __name__ == "__main__":
    model = load_model()
    model.eval()
    detector = FaceDetector()

    data_wajah = 'test_face'
    ## testing ##
    for file in os.listdir(data_wajah):
        img_path = os.path.join(data_wajah, file)
        img = cv2.imread(img_path)
        img_crop = crop_face(img)
        img_crop = load_image(img_crop)
        feature = FR_inference(img_crop)
        recog = recognition('result.json', feature)

        print(file +'='+ recog)