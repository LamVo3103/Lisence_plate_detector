import os
import time
import numpy as np
import tensorflow as tf
from model import build_lprnet
from loader import LPRDataLoader, NUM_CLASSES, INT_2_CHAR


def decode_predictions(probs):
    batch_size = tf.shape(probs)[0]
    timesteps = tf.shape(probs)[1]
    input_length = tf.fill([batch_size], timesteps)
    decodeds, _ = tf.keras.backend.ctc_decode(probs, input_length, greedy=True)
    decoded_sequences = decodeds[0].numpy()
    results = []
    for seq in decoded_sequences:
        text = ""
        for idx in seq:
            if idx != -1 and idx < NUM_CLASSES - 1:
                text += INT_2_CHAR[idx]
        results.append(text)
    return results


def train():
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 30  # Bắt đầu lại thì 30 vòng là đủ thành tài

    print("⏳ Đang nạp dữ liệu vào bộ nhớ...")
    train_loader = LPRDataLoader("./train", batch_size=BATCH_SIZE)

    # Khởi tạo Learning Rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001, decay_steps=300, decay_rate=0.9, staircase=True
    )

    model = build_lprnet(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    os.makedirs("./saved_models", exist_ok=True)
    print("🚀 Bắt đầu HUẤN LUYỆN LẠI TỪ ĐẦU (Đã vá lỗi Toán học)...")

    for epoch in range(TOTAL_EPOCHS):
        print(f"\n=== Vòng (Epoch) {epoch + 1}/{TOTAL_EPOCHS} ===")
        train_loader.shuffle_data()

        epoch_loss = 0
        num_batches = int(np.ceil(train_loader.num_samples / BATCH_SIZE))
        start_time = time.time()

        for batch_idx in range(num_batches):
            images, sparse_labels = train_loader.get_batch(batch_idx)

            sparse_labels = tf.SparseTensor(
                indices=sparse_labels[0], values=sparse_labels[1], dense_shape=sparse_labels[2]
            )

            with tf.GradientTape() as tape:
                # 1. model() xuất ra Softmax (0.0 -> 1.0)
                probs = model(images, training=True)

                # 2. VÁ LỖI: Dùng Logarit bẻ ngược Softmax về Raw Logits
                logits = tf.math.log(probs + 1e-7)
                logits_transposed = tf.transpose(logits, (1, 0, 2))

                batch_size_actual = tf.shape(logits)[0]
                timesteps = tf.shape(logits)[1]
                seq_lengths = tf.fill([batch_size_actual], timesteps)

                ctc_loss = tf.compat.v1.nn.ctc_loss(
                    labels=sparse_labels,
                    inputs=logits_transposed,
                    sequence_length=seq_lengths
                )
                loss_value = tf.reduce_mean(ctc_loss)

            grads = tape.gradient(loss_value, model.trainable_weights)

            # 3. VÁ LỖI: Cắt cụt đạo hàm nếu nó quá lớn (Gradient Clipping)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_loss += float(loss_value)

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss_value:.4f}")

        sample_probs = model(images, training=False)
        sample_preds = decode_predictions(sample_probs)
        print(f"  🔍 Mắt AI đang đọc thử: {sample_preds[:3]}")

        avg_loss = epoch_loss / num_batches
        duration = time.time() - start_time
        print(f"✅ Kết thúc Epoch {epoch + 1} - Sai số TB: {avg_loss:.4f} - Thời gian: {duration:.2f}s")

        if (epoch + 1) % 5 == 0:
            save_path = f"./saved_models/lprnet_epoch_{epoch + 1}.weights.h5"
            model.save_weights(save_path)
            print(f"💾 Đã lưu trọng số tại: {save_path}")


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    train()