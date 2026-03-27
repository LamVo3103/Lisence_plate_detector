import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import LPRNet
from loader import LPRDataset, collate_fn, NUM_CLASSES, INT_2_CHAR, BLANK_IDX


def decode_predictions(logits):
    """Dịch ma trận PyTorch ra chữ"""
    # Lấy class có xác suất cao nhất tại mỗi bước
    preds = logits.argmax(dim=2)  # Shape: (Timesteps, Batch)
    preds = preds.transpose(0, 1)  # Shape: (Batch, Timesteps)

    results = []
    for seq in preds:
        text = ""
        prev_idx = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev_idx and idx != BLANK_IDX:
                text += INT_2_CHAR[idx]
            prev_idx = idx
        results.append(text)
    return results


def train():
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 30

    # Tự động dùng Card đồ họa (GPU) nếu có, nếu không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"⏳ Đang nạp dữ liệu trên thiết bị: {device}...")
    train_dataset = LPRDataset("./train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = LPRNet(num_classes=NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    os.makedirs("./saved_models", exist_ok=True)
    print("🚀 Bắt đầu HUẤN LUYỆN bằng PyTorch...")

    for epoch in range(TOTAL_EPOCHS):
        print(f"\n=== Vòng (Epoch) {epoch + 1}/{TOTAL_EPOCHS} ===")
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)

            optimizer.zero_grad()  # Xóa gradient cũ

            logits = model(images)  # Mạng đoán ra logits
            log_probs = F.log_softmax(logits, dim=2)  # CTC PyTorch cần log_softmax

            input_lengths = torch.full(size=(images.size(0),), fill_value=logits.size(0), dtype=torch.long).to(device)

            # Tính sai số (Loss)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Lan truyền ngược và cắt xén gradient
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        scheduler.step()  # Giảm Learning rate sau mỗi epoch

        # Test đọc thử
        model.eval()
        with torch.no_grad():
            sample_logits = model(images[:3])
            sample_preds = decode_predictions(sample_logits)
            print(f"  Mắt AI đang đọc thử: {sample_preds}")

        avg_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_time
        print(f" Kết thúc Epoch {epoch + 1} - Sai số TB: {avg_loss:.4f} - Thời gian: {duration:.2f}s")

        if (epoch + 1) % 5 == 0:
            save_path = f"./saved_models/lprnet_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" Đã lưu trọng số tại: {save_path}")


if __name__ == '__main__':
    train()