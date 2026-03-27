import cv2
import easyocr
import re
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

print("Loading YOLO model...")
# Đảm bảo bạn có file .pt từ Roboflow ở đây
yolo_model = YOLO('license_plate_detector.pt')

print("Loading EasyOCR...")
# Khởi tạo EasyOCR cho tiếng Anh
reader = easyocr.Reader(['en'], gpu=False)

#Em áp dụng hậu xử lý dựa trên domain knowledge để cải thiện độ chính xác OCR.
def process_plate_text(raw_text): #sửa lại cho đúng format của biển số vn

    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    if len(cleaned_text) < 4:
        return cleaned_text

    char_list = list(cleaned_text) #chuyển thanh list

    num_map = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6', 'Q': '0'}

    char_map = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '3': 'B', '4': 'A'}

    # Luật 1: Ký tự 1 & 2 BẮT BUỘC là SỐ (Mã tỉnh)
    for i in range(min(2, len(char_list))):
        if char_list[i] in num_map:
            char_list[i] = num_map[char_list[i]]

    # Luật 2: Ký tự thứ 3 BẮT BUỘC là CHỮ CÁI (Series biển số)
    if len(char_list) > 2 and char_list[2] in char_map:
        char_list[2] = char_map[char_list[2]]

    # Luật 3: Các ký tự từ vị trí thứ 4 trở đi BẮT BUỘC là SỐ
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
        for box in boxes:# xét từng bounding box tại vì 1 ảnh có thể có nhiều biển
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            # --- Pre-processing cơ bản ---
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            large = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            # none là ko truyền kích thước cố định, fx là theo chiều ngang, fy ...
            # interpolation=cv2.INTER_CUBIC : làm ảnh mượt, nét hơn, nội quy bậc 3
            # bậc 1(nearest): copy pixel gần nhất. bậc 3(cubic) lấy 16pixel để tạo ra 1 pixel mới

            # --- ĐỌC OCR ---
            ocr_results = reader.readtext(large, detail=0, paragraph=True)
            # detail = 0 là trả về text thôi, 1 là có trả về tọa độ nữa
            #paragraph=True :  gom nhiều thành phần lại vơới nhau , cd 51f và 12344

            raw_text = "".join(ocr_results) if len(ocr_results) > 0 else "UNKNOWN"
            final_text = process_plate_text(raw_text)

            print(f"✨ Kết quả ảnh {image_path}: {final_text}")

            # Vẽ bounding box (khung màu xanh lá)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Làm nền xanh lá, chữ đen cho nổi bật
            label = final_text
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w_text, h_text), baseline = cv2.getTextSize(label, font, 1.0, 2)
            cv2.rectangle(img, (x1, y1 - h_text - 15), (x1 + w_text + 10, y1), (0, 200, 0), -1)
            cv2.putText(img, label, (x1 + 5, y1 - 8), font, 1.0, (0, 0, 0), 2)

    cv2.imshow(f"Ket qua nhan dien: {image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images = ["test_car.jpg", "test_car1.jpg", "test_car3.jpg", "test_car4.jpg", "test_car5.jpg","test_moto1.jpg","test_moto2.jpg",
              "test_car6.jpg"]
    for img_name in images:
        try:
            process_image(img_name)
        except Exception:
            pass