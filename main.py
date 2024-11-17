import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import subprocess
import os
import re
import warnings
import sys

warnings.filterwarnings("ignore")

def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        output_text.insert(tk.END, f"{script_name} 실행 중 오류 발생: {e}\n")
        return False
    return True

def read_personal_color_and_face_shape():
    try:
        with open('C:/Users/gimg6/yagomyideulong/result.txt', 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                raise ValueError("파일에 필요한 정보 부족")
            
            tone = lines[0].strip()
            face_shape = lines[1].strip()
            
            return tone, face_shape
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return None, None

def select_image(face_shape, tone):
    face_shape_to_prefix = {
        "둥근형": "circle",
        "계란형": "egg",
        "하트형": "heart",
        "육각형": "hexagon",
        "땅콩형": "peanut",
        "마름모형": "rhomb"
    }

    tone_to_suffix = {
        "봄웜톤(spring)": "spring",
        "여름쿨톤(summer)": "summer",
        "가을웜톤(fall)": "fall",
        "겨울쿨톤(winter)": "winter"
    }

    prefix = face_shape_to_prefix.get(face_shape, "unknown")
    suffix = tone_to_suffix.get(tone, "unknown")

    selected_image = f"{prefix}_{suffix}.png"
    return selected_image

def clear_result_file():
    try:
        with open('C:/Users/gimg6/yagomyideulong/result.txt', 'w') as f:
            f.write("")
    except Exception as e:
        print(f"파일 지우기 오류: {e}")

def run_makejson():
    user_name = name_entry.get()
    if not user_name:
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, "이름을 입력해주세요.\n")
        return

    output_text.delete('1.0', tk.END)

    if not run_script("makejson.py"): return
    if not run_script("personalcolor.py"): return
    if not run_script("face_categorization.py"): return
    if not run_script("similarface.py"): return

    tone, face_shape = read_personal_color_and_face_shape()
    if tone and face_shape:
        image_name = select_image(face_shape, tone)
        image_path = f'C:/Users/gimg6/yagomyideulong/character_image/{image_name}'
        
        output_text.insert(tk.END, f"{user_name}님의 퍼스널컬러 : {tone}\n")
        output_text.insert(tk.END, f"{user_name}님의 얼굴형 : {face_shape}\n")
        output_text.insert(tk.END, f"{user_name}님과 닮은 연예인 : 김가람\n")
        output_text.insert(tk.END, f"아래는 {user_name}님의 캐릭터입니다 !\n")
        
        if os.path.exists(image_path):
            display_selected_image(image_path)
        else:
            output_text.insert(tk.END, "캐릭터 이미지를 로드 불가\n")
    else:
        output_text.insert(tk.END, "캐릭터 이미지를 로드 불가\n")

    clear_result_file()

def display_selected_image(image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk
    else:
        output_text.insert(tk.END, f"이미지 파일이 존재하지 않음: {image_path}\n")

root = tk.Tk()
root.title("YAGOMYIDEULONG")
root.geometry("500x700")

name_label = tk.Label(root, text="이름:")
name_label.pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

run_button = tk.Button(root, text="실행", command=run_makejson)
run_button.pack(pady=10)

output_text = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
output_text.pack(pady=20)

image_label = tk.Label(root, width=300, height=300)
image_label.pack(pady=10)

root.mainloop()
