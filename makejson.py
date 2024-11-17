import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from itertools import compress
import json

#퍼컬 인덱스
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LEFT_EYEBROW = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_EYE = [133, 173, 157, 158, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]
RIGHT_CHEEK = [349, 348, 347, 346, 280, 425, 266, 329]
LEFT_CHEEK = [120, 119, 118, 117, 50, 205, 36, 100]

#얼굴형 인덱스
needed_indices = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    34, 227, 137, 177, 215, 138, 135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367,
    435, 401, 366, 447, 264, 116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 428, 262,
    431, 430, 434, 416, 433, 376, 352, 345
]

class DominantColors:
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))

        kmeans = KMeans(n_clusters=self.CLUSTERS, random_state=42, n_init=10)
        kmeans.fit(self.IMAGE)

        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def getHistogram(self):
        numLabels = np.arange(0, self.CLUSTERS+1)
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = self.COLORS
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        for i in range(self.CLUSTERS):
            colors[i] = colors[i].astype(int)

        fil = [colors[i][2] < 250 and colors[i][0] > 10 for i in range(self.CLUSTERS)]
        colors = list(compress(colors, fil))

        return colors, hist

def extract_dominant_colors(image, landmarks, indices):
    points = [landmarks[i] for i in indices]
    (x, y, w, h) = cv2.boundingRect(np.array(points))

    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w += 2 * padding
    h += 2 * padding

    crop = image[y:y+h, x:x+w]
    adj_points = np.array([np.array([p[0]-x, p[1]-y]) for p in points])

    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, adj_points, 1)
    mask = mask.astype(np.bool_)
    crop[np.logical_not(mask)] = [255, 0, 0]

    dom_colors = DominantColors(crop)
    colors, _ = dom_colors.getHistogram()
    return [dom_colors.rgb_to_hex(color) for color in colors]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

landmarks_list = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        original_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

                    dominant_colors = {}
                    dominant_colors['right_eyebrow'] = extract_dominant_colors(frame, landmarks, RIGHT_EYEBROW)
                    dominant_colors['left_eyebrow'] = extract_dominant_colors(frame, landmarks, LEFT_EYEBROW)
                    dominant_colors['right_eye'] = extract_dominant_colors(frame, landmarks, RIGHT_EYE)
                    dominant_colors['left_eye'] = extract_dominant_colors(frame, landmarks, LEFT_EYE)
                    dominant_colors['right_cheek'] = extract_dominant_colors(frame, landmarks, RIGHT_CHEEK)
                    dominant_colors['left_cheek'] = extract_dominant_colors(frame, landmarks, LEFT_CHEEK)

                    with open("dominant_colors.json", "w") as json_file:
                        json.dump(dominant_colors, json_file, indent=4)
                    print("dominant_colors.json 저장 성공")

                    landmarks = {}
                    for idx in needed_indices:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks[idx] = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 랜드마크 위치를 이미지에 표시
                    landmarks_list.append(landmarks)

                    with open("lab.json", "w") as f:
                        json.dump(landmarks_list, f, indent=4)
                    print("lab.json 저장 성공!")

                    cv2.imwrite('face_mesh_image.png', original_frame)
                    print("face_mesh_image.png 저장 성공!!")

            break

        cv2.imshow('Face Mesh', frame)

except Exception as e:
    print(f"오류 발생: {e}")

cap.release()
cv2.destroyAllWindows()
