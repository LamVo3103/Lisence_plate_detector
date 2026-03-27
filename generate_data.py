import os
import random
import uuid
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_WIDTH = 94
IMG_HEIGHT = 24


def generate_vietnamese_plate_text():
    provinces = [str(i).zfill(2) for i in range(11, 100)]
    letters = "ABCDEFGHKLMNPRSTUVXYZ"
    province = random.choice(provinces)
    letter = random.choice(letters)
    numbers = "".join([str(random.randint(0, 9)) for _ in range(random.choice([4, 5]))])
    return f"{province}{letter}{numbers}"


def add_noise(cv_image):
    h, w, c = cv_image.shape
    noise = np.zeros((h, w, c), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    noisy_image = cv2.addWeighted(cv_image, 0.9, noise, 0.1, 0)
    return noisy_image


def create_plate_image(text, font_path):
    canvas_w, canvas_h = IMG_WIDTH * 4, IMG_HEIGHT * 4
    img = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, 80)
    except IOError:
        print(f"Không tìm thấy font tại {font_path}!")
        exit()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_w - text_w) / 2
    y = (canvas_h - text_h) / 2 - 20

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    angle = random.uniform(-7, 7)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    if random.random() > 0.5:  # 50% cơ hội bị mờ
        k = random.choice([3, 5])
        cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)

    if random.random() > 0.7:
        cv_img = add_noise(cv_img)

    if random.random() > 0.6:
        brightness = random.uniform(0.7, 1.0)  # Làm tối ảnh đi một chút
        cv_img = (cv_img * brightness).astype(np.uint8)

    cv_img = cv2.resize(cv_img, (IMG_WIDTH, IMG_HEIGHT))

    return cv_img


def build_dataset(num_images, output_dir, font_path):
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith('.jpg'):
            os.remove(os.path.join(output_dir, f))

    print(f"Bắt đầu tạo {num_images} ảnh 'xấu xí' vào '{output_dir}'...")

    for _ in range(num_images):
        text = generate_vietnamese_plate_text()
        img = create_plate_image(text, font_path)
        filename = f"{text}_{str(uuid.uuid4())[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)


if __name__ == '__main__':
    FONT = "./plate_font.ttf"

    build_dataset(8000, "./train", FONT)
    build_dataset(1500, "./valid", FONT)
