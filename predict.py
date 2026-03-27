import cv2
import re
import warnings
import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

from model import LPRNet
from loader import CHARS, BLANK_IDX, INT_2_CHAR

print("⏳ Loading YOLO model...")
yolo_model = YOLO('license_plate_detector.pt')

print("⏳ Loading PyTorch LPRNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpr_net = LPRNet(num_classes=len(CHARS) + 1).to(device)
# Bạn nhớ train trước để sinh ra file .pth nhé
# lpr_net.load_state_dict(torch.load('saved_models/lprnet_epoch_30.pth', map_location=device))
lpr_net.eval()

known_plate_errors = {"8": "3", "Q": "0"}


def decode_ctc(logits):
    """Dịch logits PyTorch ra chữ"""
    preds = logits.argmax(dim=2).squeeze(1)  # Lấy nhãn cao nhất
    raw_text = ""
    prev_idx = -1
    for idx in preds:
        idx = idx.item()
        if idx != prev_idx and idx != BLANK_IDX:
            raw_text += INT_2_CHAR[idx]
        prev_idx = idx
    return raw_text


def apply_lprnet_context(raw_text):
    """Hậu xử lý bằng Tri thức chuyên gia Việt Nam"""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if len(cleaned) < 4: return cleaned
    char_list = list(cleaned)

    for i in range(min(2, len(char_list))):
        if char_list[i] in known_plate_errors:
            char_list[i] = known_plate_errors[char_list[i]]

    num_map = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6', 'Q': '0'}
    for i in range(min(2, len(char_list))):
        if char_list[i] in num_map: char_list[i] = num_map[char_list[i]]

    alpha_map = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '3': 'B', '4': 'A'}
    if len(char_list) > 2 and char_list[2] in alpha_map:
        char_list[2] = alpha_map[char_list[2]]

    for i in range(3, len(char_list)):
        if char_list[i] in num_map: char_list[i] = num_map[char_list[i]]

    return "".join(char_list)


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return

    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            h_plate, w_plate = plate_crop.shape[:2]
            ch, cw = int(h_plate * 0.02), int(w_plate * 0.02)
            plate_clean = plate_crop[ch:h_plate - ch, cw:w_plate - cw]
            aspect_ratio = w_plate / h_plate

            def get_text_from_lprnet(img_crop):
                """Chuyển ảnh cho mô hình PyTorch đoán"""
                img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img_rgb, (94, 24))

                # PyTorch yêu cầu (Batch, C, H, W)
                img_input = img_resize.astype('float32') / 255.0
                img_input = np.transpose(img_input, (2, 0, 1))
                img_tensor = torch.tensor(img_input).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = lpr_net(img_tensor)

                return decode_ctc(logits)

            # --- SMART ROUTING & CHIA ĐỂ TRỊ ---
            if aspect_ratio > 2.0:
                raw_full = get_text_from_lprnet(plate_clean)
            else:
                gray_for_split = cv2.cvtColor(plate_clean, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_for_split, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                horizontal_projection = np.sum(binary, axis=1)

                mid_h = gray_for_split.shape[0] // 2
                search_range = int(gray_for_split.shape[0] * 0.2)
                search_min = max(0, mid_h - search_range)
                search_max = min(gray_for_split.shape[0], mid_h + search_range)
                split_y = search_min + np.argmin(horizontal_projection[search_min:search_max])

                text_top = get_text_from_lprnet(plate_clean[0:split_y, :])
                text_bottom = get_text_from_lprnet(plate_clean[split_y:, :])
                raw_full = text_top + text_bottom

            final_text = apply_lprnet_context(raw_full)
            print(f"✨ Kết quả ảnh {image_path}: {final_text}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w_text, h_text), _ = cv2.getTextSize(final_text, font, 1.0, 2)
            cv2.rectangle(img, (x1, y1 - h_text - 15), (x1 + w_text + 10, y1), (40, 40, 40), -1)
            cv2.putText(img, final_text, (x1 + 5, y1 - 8), font, 1.0, (255, 255, 255), 2)

    cv2.imshow(f"Ket qua nhan dien: {image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images = ["test_car.jpg", "test_car1.jpg", "test_car2.jpg", "test_car3.jpg", "test_car4.jpg", "test_car5.jpg",
              "test_car6.jpg"]
    for img_name in images:
        try:
            process_image(img_name)
        except Exception:
            pass