import tensorflow as tf
from tensorflow.keras import layers, Model


def conv2d_batchnorm(x, filters, kernel_size, strides=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def small_basic_block(x, filters_out):
    filters_in = filters_out // 4
    x1 = conv2d_batchnorm(x, filters_in, (1, 1))
    x2 = conv2d_batchnorm(x1, filters_in, (3, 1))
    x3 = conv2d_batchnorm(x2, filters_in, (1, 3))
    x4 = conv2d_batchnorm(x3, filters_out, (1, 1))
    return x4


def build_lprnet(input_shape=(24, 94, 3), num_classes=32):

    inputs = layers.Input(shape=input_shape)

    # 1. Khối Input
    x = conv2d_batchnorm(inputs, 64, (3, 3))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = small_basic_block(x, 128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 1), padding='same')(x)

    x = small_basic_block(x, 256)
    x = small_basic_block(x, 256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 1), padding='same')(x)

    x = layers.Dropout(0.5)(x)
    x = conv2d_batchnorm(x, 256, (4, 1), padding='valid')
    x = layers.Dropout(0.5)(x)
    classes = conv2d_batchnorm(x, num_classes, (1, 13), padding='same')

    pattern = layers.Conv2D(128, (1, 1))(classes)
    x = layers.Concatenate()([classes, pattern])
    x = conv2d_batchnorm(x, num_classes, (1, 1), padding='same')

    x = layers.Lambda(lambda tensor: tf.reduce_mean(tensor, axis=1))(x)

    outputs = layers.Activation('softmax', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="LPRNet")
    return model


if __name__ == '__main__':
    model = build_lprnet()
    model.summary()