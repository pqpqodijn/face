import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
from scipy.spatial.distance import euclidean
import numpy as np

mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

#얼굴 임베딩
def get_embedding(image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    if img_cropped is None or len(img_cropped) == 0:
        raise ValueError("No faces detected in the image")
    img_cropped = img_cropped[0].unsqueeze(0)
    embedding = model(img_cropped)
    return embedding

#두 임베딩 벡터 사이의 유클리드 거리 계산
def calculate_distance(embedding1, embedding2):
    embedding1 = embedding1.detach().numpy().flatten()
    embedding2 = embedding2.detach().numpy().flatten()
    distance = euclidean(embedding1, embedding2)
    return distance

def find_most_similar_image(input_image_path, database_image_paths):
    input_embedding = get_embedding(input_image_path)

    min_distance = float('inf')
    most_similar_image = None

    for image_path in database_image_paths:
        try:
            db_embedding = get_embedding(image_path)
        except ValueError:
            print(f"얼굴 감지 불가! : {image_path}")
            continue

        distance = calculate_distance(input_embedding, db_embedding)

        if distance < min_distance:
            min_distance = distance
            most_similar_image = image_path

    return most_similar_image, min_distance

database_dir = 'C:/Users/gimg6/yagomyideulong/data_celebrity'
database_image_paths = [os.path.join(database_dir, f) for f in os.listdir(database_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
input_image_path = os.path.abspath('C:/Users/gimg6/yagomyideulong/face_mesh_image.png')

try:
    most_similar_image, min_distance = find_most_similar_image(input_image_path, database_image_paths)
    print(f'가장 비슷한 이미지: {most_similar_image}')
    print(f'거리차: {min_distance}')
except ValueError as e:
    print(f"얼굴 감지 불가! : {input_image_path}")
