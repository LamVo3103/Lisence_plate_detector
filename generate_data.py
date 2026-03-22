import os
import random
import uuid
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Kích thước chuẩn đầu vào của mạng LPRNet
IMG_WIDTH = 94
IMG_HEIGHT = 24


def generate_vietnamese_plate_text():
    """Hàm tạo chuỗi biển số xe ô tô ngẫu nhiên (VD: 51A12345)"""
    provinces = [str(i).zfill(2) for i in range(11, 100)]
    letters = "ABCDEFGHKLMNPRSTUVXYZ"
    province = random.choice(provinces)
    letter = random.choice(letters)
    numbers = "".join([str(random.randint(0, 9)) for _ in range(random.choice([4, 5]))])
    return f"{province}{letter}{numbers}"


def add_noise(cv_image):
    """Thêm nhiễu hạt tiêu/muối để ảnh bớt sạch"""
    h, w, c = cv_image.shape
    noise = np.zeros((h, w, c), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    # Giảm cường độ nhiễu xuống để không che mất chữ
    noisy_image = cv2.addWeighted(cv_image, 0.9, noise, 0.1, 0)
    return noisy_image


def create_plate_image(text, font_path):
    """Vẽ chữ biển số với Heavy Augmentation"""
    # 1. Tạo nền trắng to để xoay không bị mất góc
    canvas_w, canvas_h = IMG_WIDTH * 4, IMG_HEIGHT * 4
    img = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Sử dụng font size to để vẽ cho nét
    try:
        font = ImageFont.truetype(font_path, 80)
    except IOError:
        print(f"Không tìm thấy font tại {font_path}!")
        exit()

    # Căn giữa chữ
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_w - text_w) / 2
    y = (canvas_h - text_h) / 2 - 20

    # Vẽ chữ màu đen
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # 2. AUGMENTATION 1: Xoay ngẫu nhiên (Skew) từ -7 đến 7 độ
    angle = random.uniform(-7, 7)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # Chuyển ảnh PIL sang OpenCV (Numpy)
    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # 3. AUGMENTATION 2: Làm mờ ngẫu nhiên (Blur)
    if random.random() > 0.5:  # 50% cơ hội bị mờ
        k = random.choice([3, 5])
        cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)

    # 4. AUGMENTATION 3: Thêm nhiễu hạt (Noise)
    if random.random() > 0.7:
        cv_img = add_noise(cv_img)

    # 5. AUGMENTATION 4: Giả lập bóng đổ nhẹ (Intensity change)
    if random.random() > 0.6:
        brightness = random.uniform(0.7, 1.0)  # Làm tối ảnh đi một chút
        cv_img = (cv_img * brightness).astype(np.uint8)

    # Bóp về đúng chuẩn LPRNet
    cv_img = cv2.resize(cv_img, (IMG_WIDTH, IMG_HEIGHT))

    return cv_img


def build_dataset(num_images, output_dir, font_path):
    os.makedirs(output_dir, exist_ok=True)
    # Xóa ảnh cũ trong thư mục để tránh bị lẫn dữ liệu
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
    # TRỎ ĐÚNG ĐƯỜNG DẪN TỚI FILE FONT CỦA BẠN (Dùng Arial Bold hoặc UKNumberPlate đều được)
    FONT = "./plate_font.ttf"

    # Tạo lại dữ liệu với số lượng nhiều hơn để mô hình học giỏi hơn
    build_dataset(8000, "./train", FONT)  # Tăng lên 8000 ảnh Train
    build_dataset(1500, "./valid", FONT)  # Tăng lên 1500 ảnh Valid

    print(" Tuyệt vời! Đã hoàn tất việc tạo Dataset 'Khắc nghiệt'.")