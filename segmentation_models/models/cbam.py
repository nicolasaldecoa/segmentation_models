__all__ = ["CBAM", "channelwise_avg_pooling", "channelwise_max_pooling"]

from keras import backend
from keras import layers


def channelwise_avg_pooling(x):
    return backend.mean(x, axis=3, keepdims=True)


def channelwise_max_pooling(x):
    return backend.max(x, axis=3, keepdims=True)


def CBAM(ratio=8, kernel_size=7, name=None):
    """
    Convolutional Block Attention Module (CBAM)
    as described in https://arxiv.org/abs/1807.06521
    """

    def wrapper(input_feature):

        # CHANNEL ATTENTION
        channels = backend.int_shape(input_feature)[-1]

        shared_layer_one = layers.Dense(max(1, channels // ratio),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        use_bias=True,
                                        bias_initializer='zeros',
                                        name=name + "_c_hidden" if name is not None else None)
        shared_layer_two = layers.Dense(channels,
                                        kernel_initializer='he_normal',
                                        use_bias=True,
                                        bias_initializer='zeros',
                                        name=name + "_c_weights" if name is not None else None)

        avg_pool = layers.GlobalAveragePooling2D(name=name + "_c_avg" if name is not None else None)(input_feature)
        avg_pool = layers.Reshape((1, 1, channels),
                                  name=name + "_c_avg_flat" if name is not None else None)(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        max_pool = layers.GlobalMaxPooling2D(name=name + "_c_max" if name is not None else None)(input_feature)
        max_pool = layers.Reshape((1, 1, channels),
                                  name=name + "_c_max_flat" if name is not None else None)(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        cbam_feature = layers.Add(name=name + "_c_avgmax")([avg_pool, max_pool])
        cbam_feature = layers.Activation('sigmoid',
                                         name=name + "_c_sigmoid" if name is not None else None)(cbam_feature)

        cbam_feature = layers.Multiply(name=name + "_c_rescale" if name is not None else None)([input_feature,
                                                                                                cbam_feature])

        # SPATIAL ATTENTION
        avg_pool = layers.Lambda(channelwise_avg_pooling,
                                 name=name + "_s_avg" if name is not None else None)(cbam_feature)
        max_pool = layers.Lambda(channelwise_max_pooling,
                                 name=name + "_s_max" if name is not None else None)(cbam_feature)
        concat = layers.Concatenate(axis=3, name=name + "_s_avgmax" if name is not None else None)([avg_pool, max_pool])
        attention = layers.Conv2D(filters=1,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False,
                                  name=name + "_s_conv" if name is not None else None)(concat)

        return layers.Multiply(name=name + "_s_rescale" if name is not None else None)([cbam_feature, attention])

    return wrapper
