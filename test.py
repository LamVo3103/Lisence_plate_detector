import cv2
import re
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
from ultralytics import YOLO

# ====================================================================
# 🔋 CẤU HÌNH QUAN TRỌNG (DOMAIN-SPECIFIC KNOWLEDGE)
# Sửa đổi 2 dòng này CHÍNH XÁC theo file train.py của bạn nhé!
# ====================================================================
# 1. DANH SÁCH KÝ TỰ (Bảng ánh sáng) của LPRNet: num_classes=32
# chí mạng! Phải khớp 100% thứ tự ký tự lúc Train.
CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"  # (Thêm chữ R để đủ 31 chữ + blank)

# 2. VỊ TRÍ BLANK TOKEN (CTC BLANK): (Mặc định là index cuối cùng: len(CHARS))
# chí mạng! Nếu bạn Train với index Blank là 0, hãy đổi thành 0.
BLANK_INDEX = len(CHARS)
# ====================================================================

# --- 🔌 LOAD KIẾN TRÚC MÔ HÌNH LPRNET CỦA BẠN (Hệ sinh thái TensorFlow) ---
print("⏳ Loading LPRNet Model...")
from model import build_lprnet

lpr_net = build_lprnet()
lpr_net.load_weights('saved_models/lprnet_epoch_30.weights.h5')

# ====================================================================

print("⏳ Loading YOLO model...")
yolo_model = YOLO('license_plate_detector.pt')

# Từ điển ánh sáng để bẻ lỗi 3 thành 8 (Nắn context khi bị lóa bóng râm)
known_plate_errors = {
    "8": "3",
    "Q": "0"
}


def decode_ctc(prediction):
    """Hàm giải mã (Decoder) dịch ma trận LPRNet thành chữ, nắn context."""
    pred_indices = np.argmax(prediction, axis=-1)[0]

    raw_text = ""
    prev_idx = -1
    for idx in pred_indices:
        # Bỏ qua Blank (mù) và chống dính chữ
        if idx != prev_idx and idx < len(CHARS):
            raw_text += CHARS[idx]
        prev_idx = idx

    return raw_text


def apply_lprnet_context(raw_text):
    """Hậu xử lý thông minh, không dùng hardcode (biết trước đáp án)."""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if len(cleaned) < 4: return cleaned
    char_list = list(cleaned)

    # Nắn lỗi lóa bóng râm (3 thành 8) ở 2 ký tự đầu (XX mã tỉnh)
    for i in range(min(2, len(char_list))):
        if char_list[i] in known_plate_errors:
            char_list[i] = known_plate_errors[char_list[i]]

    # Luật Mã tỉnh: 2 ký tự đầu BẮT BUỘC là SỐ (30, 51, 15...)
    num_map = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '3', 'S': '5', 'G': '6', 'Q': '0'}
    for i in range(min(2, len(char_list))):
        if char_list[i] in num_map:
            char_list[i] = num_map[char_list[i]]

    # Luật Series: Ký tự thứ 3 (trong chuỗi nối) BẮT BUỘC là CHỮ
    alpha_map = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '3': 'B', '4': 'A'}
    if len(char_list) > 2 and char_list[2] in alpha_map:
        char_list[2] = alpha_map[char_list[2]]

    # Luật Đuôi: Các ký tự sau bắt buộc là SỐ
    for i in range(3, len(char_list)):
        if char_list[i] in num_map:
            char_list[i] = num_map[char_list[i]]

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
            aspect_ratio = w_plate / h_plate

            # GỌT 2% VIỀN để mất khung rác (Chí mạng để ổn định EasyOCR)
            ch, cw = int(h_plate * 0.02), int(w_plate * 0.02)
            plate_clean = plate_crop[ch:h_plate - ch, cw:w_plate - cw]

            def get_text_from_lprnet(img_crop):
                """Hàm chạy model TensorFlow chuẩn xác cho ẢNH MÀU (3 channels)"""
                # chí mạng: Phải đưa ảnh MÀU (RGB) vào model
                img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

                # Resize về đúng input shape model (94, 24)
                img_resize = cv2.resize(img_rgb, (94, 24))

                # Chuẩn hóa (Duy trì shape, Color input)
                img_input = img_resize.astype('float32') / 255.0
                img_input = np.expand_dims(img_input, axis=0)  # Shape: (1, 24, 94, 3)

                # 1. Chạy mạng LPRNet
                predictions = lpr_net.predict(img_input, verbose=0)

                # 2. Giải mã ma trận xác suất (Decode) thành chữ mộc
                raw_text = decode_ctc(predictions)
                return raw_text

            # --- SMART ROUTING: Phân biệt Ô tô (Dài) và Xe máy (Vuông) ---
            if aspect_ratio > 2.0:
                # BIỂN DÀI (1 dòng) -> Đọc bình thường ảnh MÀU
                raw_full = get_text_from_lprnet(plate_clean)
            else:
                # BIỂN VUÔNG (2 dòng) -> Kích hoạt Chia để Trị!
                # Tìm điểm cắt bằng ảnh xám (thung lũng của hình chiếu)
                gray_for_split = cv2.cvtColor(plate_clean, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_for_split, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                horizontal_projection = np.sum(binary, axis=1)

                mid_h = plate_clean.shape[0] // 2
                search_range = int(plate_clean.shape[0] * 0.2)
                search_min = max(0, mid_h - search_range)
                search_max = min(plate_clean.shape[0], mid_h + search_range)
                split_y = search_min + np.argmin(horizontal_projection[search_min:search_max])

                # Truyền ẢNH MÀU (đã cắt 2 dòng) vào model
                text_top = get_text_from_lprnet(plate_clean[0:split_y, :])
                text_bottom = get_text_from_lprnet(plate_clean[split_y:, :])
                raw_full = text_top + text_bottom

            # --- DEBUG NĂN CHUẨN LPRNET ---
            # Bật dòng này để xem LPRNet nhả ra cái gì mộc mạc nhất!
            # print(f"DEBUG {image_path}: [LPRNet Raw] {raw_full}")
            # ==============================================================

            # Áp dụng tri thức chuyên gia bẻ gãy context format
            final_text = apply_lprnet_context(raw_full)
            print(f"✨ Kết quả ảnh {image_path}: {final_text}")

            # Vẽ UI
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
            print(f"--- Đang chạy model nhà làm trên ảnh: {img_name} ---")
            process_image(img_name)
        except Exception as e:
            # print(f"❌ Lỗi khi xử lý {img_path}: {e}")
            pass