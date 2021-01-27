from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file


def vgg16(img_input=(224, 224, 3), path_url=None, path_to_save_h5_file=None):
    """

    :param path_to_save_h5_file: Destination directory to store a downloaded h5 file with a preferred name. (type-> str)
    :param img_input: size of the image as tuple [(224,224,3)--> Default]. (type-> tuple)
    :param path_url: url_path to the pretrained weights. (type-> str)
    :return: return the vgg_16 model without including top

    :example

    models = vgg16(img_input=(560, 560, 3),
               path_url='https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
               path_to_save_h5_file='/media/sri/shared/SEM-8/cassava leaf disease classification/vgg16_without_top.h5')

    """

    # Block 1
    input = Input(shape=img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(input, x, name='vgg16')
    path = get_file(path_to_save_h5_file, path_url,
                    cache_dir='/media/sri/shared/SEM-8/cassava leaf disease classification/cache')

    model.load_weights(path)
    model.summary()

    return model


# try
'''
models = vgg16(img_input=(560, 560, 3),
               path_url='https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
               path_to_save_h5_file='/media/sri/shared/SEM-8/cassava leaf disease classification/vgg16_without_top.h5')
'''