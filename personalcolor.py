import json
import numpy as np
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color

def is_warm(lab_b, a):
    warm_b_std = [11.6518, 11.71445, 3.6484]
    cool_b_std = [4.64255, 4.86635, 0.18735]

    warm_dist = sum(abs(lab_b[i] - warm_b_std[i]) * a[i] for i in range(3))
    cool_dist = sum(abs(lab_b[i] - cool_b_std[i]) * a[i] for i in range(3))

    return 1 if warm_dist <= cool_dist else 0

def is_spr(hsv_s, a):
    spr_s_std = [18.59296, 30.30303, 25.80645]
    fal_s_std = [27.13987, 39.75155, 37.5]

    spr_dist = sum(abs(hsv_s[i] - spr_s_std[i]) * a[i] for i in range(3))
    fal_dist = sum(abs(hsv_s[i] - fal_s_std[i]) * a[i] for i in range(3))

    return 1 if spr_dist <= fal_dist else 0

def is_smr(hsv_s, a):
    smr_s_std = [12.5, 21.7195, 24.77064]
    wnt_s_std = [16.73913, 24.8276, 31.3726]
    a[1] = 0.5  # eyebrow 가중치 줄임!

    smr_dist = sum(abs(hsv_s[i] - smr_s_std[i]) * a[i] for i in range(3))
    wnt_dist = sum(abs(hsv_s[i] - wnt_s_std[i]) * a[i] for i in range(3))

    return 1 if smr_dist <= wnt_dist else 0

def calculate_personal_color(colors_json_path):
    with open(colors_json_path, 'r') as file:
        dominant_colors = json.load(file)

    #얼굴 부위별로 색깔 평균 계산
    color_parts = ['right_cheek', 'left_cheek', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye']
    colors = {}

    for part in color_parts:
        if 'cheek' in part:
            key = 'cheek'
        elif 'eyebrow' in part:
            key = 'eyebrow'
        elif 'eye' in part:
            key = 'eye'

        hex_color = dominant_colors[part][0]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]

        if key not in colors:
            colors[key] = []
        colors[key].append(rgb_color)

    cheek_avg = np.mean(colors['cheek'], axis=0)
    eyebrow_avg = np.mean(colors['eyebrow'], axis=0)
    eye_avg = np.mean(colors['eye'], axis=0)

    #색상 변환, HSV 색상 추출
    color_avg = [cheek_avg, eyebrow_avg, eye_avg]
    Lab_b, hsv_s = [], []

    for rgb in color_avg:
        rgb = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
        lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)
        hsv = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor)
        Lab_b.append(float(format(lab.lab_b, ".2f")))
        hsv_s.append(float(format(hsv.hsv_s, ".2f")) * 100)

   
    #퍼컬 분석
    Lab_weight = [30, 20, 5]
    hsv_weight = [10, 1, 1]

    if is_warm(Lab_b, Lab_weight):
        if is_spr(hsv_s, hsv_weight):
            tone = '봄웜톤(spring)'
        else:
            tone = '가을웜톤(fall)'
    else:
        if is_smr(hsv_s, hsv_weight):
            tone = '여름쿨톤(summer)'
        else:
            tone = '겨울쿨톤(winter)'

    try:
        with open('C:/Users/gimg6/yagomyideulong/result.txt', 'w') as f:
            f.write(f"{tone}\n")
            print(f"{tone}작성되었습니다")
    except Exception as e:
        print(f"파일 쓰기 오류: {e}")

    print(f"당신의 퍼스널 컬러는 {tone}입니다.")


colors_json_path = 'C:/Users/gimg6/yagomyideulong/dominant_colors.json'

calculate_personal_color(colors_json_path)
