import tensorflow as tf
from tensorflow.keras import layers, Model


def conv2d_batchnorm(x, filters, kernel_size, strides=1, padding='same'):
    """Khối cơ bản: Tích chập (Conv2D) -> Chuẩn hóa (BatchNorm) -> Kích hoạt (ReLU)"""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def small_basic_block(x, filters_out):
    """Khối Basic Block giúp tăng tốc độ xử lý bằng cách chia nhỏ bộ lọc"""
    filters_in = filters_out // 4
    x1 = conv2d_batchnorm(x, filters_in, (1, 1))
    x2 = conv2d_batchnorm(x1, filters_in, (3, 1))
    x3 = conv2d_batchnorm(x2, filters_in, (1, 3))
    x4 = conv2d_batchnorm(x3, filters_out, (1, 1))
    return x4


def build_lprnet(input_shape=(24, 94, 3), num_classes=32):
    """
    Xây dựng toàn bộ kiến trúc LPRNet
    num_classes = 31 (chữ/số) + 1 (ký tự Blank cho CTC Loss)
    """
    inputs = layers.Input(shape=input_shape)

    # 1. Khối Input
    x = conv2d_batchnorm(inputs, 64, (3, 3))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = small_basic_block(x, 128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 1), padding='same')(x)

    # 2. Khối Basic và Convolution
    x = small_basic_block(x, 256)
    x = small_basic_block(x, 256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 1), padding='same')(x)

    # 3. Khối Dropout chống "học vẹt" (Overfitting)
    x = layers.Dropout(0.5)(x)
    x = conv2d_batchnorm(x, 256, (4, 1), padding='valid')
    x = layers.Dropout(0.5)(x)

    # 4. Khối Tiên đoán (Head)
    classes = conv2d_batchnorm(x, num_classes, (1, 13), padding='same')

    # Kết hợp thêm bối cảnh toàn cục (Global Context)
    pattern = layers.Conv2D(128, (1, 1))(classes)
    x = layers.Concatenate()([classes, pattern])
    x = conv2d_batchnorm(x, num_classes, (1, 1), padding='same')

    # 5. Ép dẹt chiều cao (Reduce Mean) để tạo thành dạng chuỗi 1D
    # Từ shape (Batch, Chiều_cao, Chiều_rộng, Số_lớp) -> (Batch, Chiều_rộng, Số_lớp)
    x = layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1))(x)

    # Lớp Softmax để tính xác suất cho từng chữ cái
    outputs = layers.Activation('softmax', dtype='float32')(x)

    # Đóng gói thành Model
    model = Model(inputs=inputs, outputs=outputs, name="LPRNet")
    return model


if __name__ == '__main__':
    # Chạy thử file này để xem cấu trúc mạng đã dựng thành công chưa
    model = build_lprnet()
    model.summary()