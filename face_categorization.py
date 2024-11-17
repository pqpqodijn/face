import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def extract_features(landmarks):
    """특징 벡터 추출"""
    points = [(landmarks[key]['x'], landmarks[key]['y'], landmarks[key]['z']) for key in landmarks]
    features = np.array(points).flatten()
    return features

def get_landmarks_from_json(json_path):
    """JSON 파일 불러오는 중!"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'landmarks' in data:
        landmarks_list = data['landmarks']
    elif isinstance(data, list):
        landmarks_list = data
    else:
        raise ValueError(f"json형식 이상함 {json_path}")

    return landmarks_list

def get_features_from_json(json_path):
    """얼굴 랜드마크 특징 벡터를 추출"""
    landmarks_list = get_landmarks_from_json(json_path)
    if not landmarks_list:
        raise ValueError(f"해당 랜드마크가 없음 {json_path}")
    landmarks = landmarks_list[0] 
    features = extract_features(landmarks)
    return features

def prepare_training_data(label_to_paths):

    features = []
    all_labels = []
    for label, json_paths in label_to_paths.items():
        for json_path in json_paths:
            features.append(get_features_from_json(json_path))
            all_labels.append(label)
    return np.array(features), np.array(all_labels)

def classify_face_shape(new_image_json_path, label_to_paths, k=5):
    """이미지 분류"""
    X_train, y_train = prepare_training_data(label_to_paths)

    #데이터 정규화 파트
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
   
    new_features = get_features_from_json(new_image_json_path)
    new_features_scaled = scaler.transform([new_features])
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    predicted_label = knn.predict(new_features_scaled)
    try:
        with open('C:/Users/gimg6/yagomyideulong/result.txt', 'a') as f:
            f.write(predicted_label[0])
            print(f"{predicted_label[0]}저장됨")
    except Exception as e:
        print(f"파일 작성 오류! : {e}")
    
    return predicted_label[0]
    

label_to_paths = {
    '둥근형': [
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle2.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle4.json',
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle8.json',
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/circle/circle10.json'
    ],
    '계란형': [
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg2.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg4.json',
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg8.json',
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/egg/egg10.json'
    ],
    '하트형': [
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart2.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart4.json',
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart8.json',
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/heart/heart10.json'
    ],
    '육각형': [
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon2.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon4.json',
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon8.json',
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/hexagon/hexagon10.json'
    ],
    '땅콩형': [
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut4.json',
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut8.json',
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/peanut/peanut10.json'
    ],
    '마름모형': [
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb1.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb2.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb3.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb4.json',
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb5.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb6.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb7.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb8.json',
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb9.json', 
        'C:/Users/gimg6/yagomyideulong/datas/rhomb/rhomb10.json'
    ]
}

new_image_json_path = 'C:/Users/gimg6/yagomyideulong/lab.json'
predicted_face_shape = classify_face_shape(new_image_json_path, label_to_paths)

print(f"얼굴형: {predicted_face_shape}")
