#Chịu trách nhiệm đọc ảnh từ thư mục và ép kiểu ma trận để đưa vào mạng Neural.

import os
import cv2
import numpy as np

IMG_WIDTH = 94
IMG_HEIGHT = 24

CHARS = "ABCDEFGHKLMNPRSTUVXYZ0123456789"
NUM_CLASSES = len(CHARS) + 1

CHAR_2_INT = {char: i for i, char in enumerate(CHARS)}
INT_2_CHAR = {i: char for i, char in enumerate(CHARS)}

def encode_label(label_str):    # chuyển chuỗi thành số
    return [CHAR_2_INT[c] for c in label_str]


def get_sparse_tuple_from(sequences, dtype=np.int32):
    """
    Hàm đặc thù để tạo Sparse Tensor.
    Mạng LPRNet dùng hàm tính lỗi CTC Loss, hàm này bắt buộc
    nhãn đầu vào phải ở dạng ma trận thưa (Sparse).
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


class LPRDataLoader:
    """Lớp chịu trách nhiệm bốc dỡ và nhào nặn dữ liệu để đút cho AI học"""

    def __init__(self, img_dir, batch_size=32):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        #trộn ngẫu nhiên để tránh học vẹt
        np.random.shuffle(self.filenames)
        self.num_samples = len(self.filenames)

    def get_batch(self, batch_idx):
        """Lấy 1 cụm (batch) dữ liệu để đưa vào mô hình"""
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)

        batch_filenames = self.filenames[start_idx:end_idx]

        # Khởi tạo ma trận rỗng chứa Ảnh và Nhãn
        images = np.zeros([len(batch_filenames), IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.float32)
        labels = []

        for i, fname in enumerate(batch_filenames):
            # --- XỬ LÝ ẢNH ---
            img_path = os.path.join(self.img_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # Chuẩn hóa màu ảnh (Chia 255 để đưa giá trị pixel về khoảng 0.0 -> 1.0)
            img = img.astype(np.float32) / 255.0
            images[i] = img

            label_str = fname.split('_')[0]
            labels.append(encode_label(label_str))

        # Đóng gói nhãn thành Sparse Tensor
        sparse_labels = get_sparse_tuple_from(labels)

        return images, sparse_labels

    def shuffle_data(self):
        # trộn
        np.random.shuffle(self.filenames)