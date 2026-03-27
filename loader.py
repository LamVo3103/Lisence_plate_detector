import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_WIDTH = 94
IMG_HEIGHT = 24

CHARS = "ABCDEFGHKLMNPRSTUVXYZ0123456789"
NUM_CLASSES = len(CHARS) + 1  # Cộng 1 cho ký tự Blank của CTC
BLANK_IDX = NUM_CLASSES - 1

CHAR_2_INT = {char: i for i, char in enumerate(CHARS)}
INT_2_CHAR = {i: char for i, char in enumerate(CHARS)}


class LPRDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # PyTorch chuộng RGB
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Chuẩn hóa về [0, 1]
        img = img.astype(np.float32) / 255.0

        # CHUYỂN ĐỔI CHÍ MẠNG: Keras (H, W, C) -> PyTorch (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # Tách nhãn từ tên file
        label_str = fname.split('_')[0]
        label = [CHAR_2_INT[c] for c in label_str]

        return torch.tensor(img), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Hàm gom nhóm dữ liệu đặc biệt cho CTC Loss của PyTorch"""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    # CTC Loss PyTorch yêu cầu 1 mảng 1D chứa TẤT CẢ ký tự và mảng độ dài
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)

    return images, targets, target_lengths