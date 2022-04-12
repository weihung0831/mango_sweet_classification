import pretty_errors
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, BatchNormalization, Lambda, \
    Concatenate, Convolution2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def load_path(label_path, label_num):
    image = []
    label = []

    for files in os.listdir(label_path):
        path = os.path.join(label_path, files)
        rgbd_img = np.load(path, encoding='bytes', allow_pickle=True)
        rgbd_img = rgbd_img['rgbd']

        image.append(rgbd_img)
        label.append(label_num)

    image = np.array(image)
    label = np.array(label)

    # print(image.shape)

    return image, label


def onehot_encoder(label):
    onehot = OneHotEncoder()
    onehot.fit(label)
    onehot = onehot.transform(label).toarray()

    return onehot


def sequential():
    model = Sequential()
    model.add(Conv2D(input_shape=(100, 100, 4), filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    return model


def vgg19():
    model = Sequential()
    model.add(Conv2D(input_shape=(100, 100, 4), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()

    # img = tf.keras.utils.plot_model(model, to_file='model/sweet_model/model.png', show_shapes=True)

    return model


def color_classification():
    # placeholder for input image
    input_image = Input(shape=(100, 100, 4))
    # ============================================= TOP BRANCH ===================================================
    # first top convolution layer
    top_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                              input_shape=(224, 224, 3), activation='relu')(input_image)
    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_conv1)

    # second top convolution layer
    # split feature map by half
    top_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(top_conv1)
    top_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(top_conv1)

    top_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv2)
    top_top_conv2 = BatchNormalization()(top_top_conv2)
    top_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv2)

    top_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv2)
    top_bot_conv2 = BatchNormalization()(top_bot_conv2)
    top_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv2)

    # third top convolution layer
    # concat 2 feature map
    top_conv3 = Concatenate()([top_top_conv2, top_bot_conv2])
    top_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_conv3)

    # fourth top convolution layer
    # split feature map by half
    top_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(top_conv3)
    top_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(top_conv3)

    top_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)
    top_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)

    # fifth top convolution layer
    top_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)
    top_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv5)

    top_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)
    top_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv5)

    # ============================================= TOP BOTTOM ===================================================
    # first bottom convolution layer
    bottom_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                                 input_shape=(224, 224, 3), activation='relu')(input_image)
    bottom_conv1 = BatchNormalization()(bottom_conv1)
    bottom_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_conv1)

    # second bottom convolution layer
    # split feature map by half
    bottom_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(bottom_conv1)
    bottom_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(bottom_conv1)

    bottom_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv2)
    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
    bottom_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv2)

    bottom_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv2)
    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
    bottom_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv2)

    # third bottom convolution layer
    # concat 2 feature map
    bottom_conv3 = Concatenate()([bottom_top_conv2, bottom_bot_conv2])
    bottom_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_conv3)

    # fourth bottom convolution layer
    # split feature map by half
    bottom_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(bottom_conv3)
    bottom_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(bottom_conv3)

    bottom_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)
    bottom_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)

    # fifth bottom convolution layer
    bottom_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)
    bottom_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv5)

    bottom_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)
    bottom_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv5)

    # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
    conv_output = Concatenate()([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5])

    # Flatten
    flatten = Flatten()(conv_output)

    # Fully-connected layer
    FC_1 = Dense(units=4096, activation='relu')(flatten)
    FC_1 = Dropout(0.6)(FC_1)
    FC_2 = Dense(units=4096, activation='relu')(FC_1)
    FC_2 = Dropout(0.6)(FC_2)
    output = Dense(units=1, activation='sigmoid')(FC_2)

    model = Model(inputs=input_image, outputs=output)
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
    model.compile(optimizer=sgd, loss='BinaryCrossentropy', metrics=['accuracy'])

    model.summary()

    return model


def compile(model, x, y):
    # lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100, decay_rate=0.95, staircase=False)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='BinaryCrossentropy', metrics=['accuracy'])
    # early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=0, mode='max',
    #                                baseline=None, restore_best_weights=True)
    history = model.fit(x, y, epochs=150, batch_size=8, validation_split=0.25)

    return history


def training_model_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.title('Training and Validation Accuracy')
    plt.grid(linestyle=':')
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])

    plt.subplot(212)
    plt.title('Training and Validation Loss')
    plt.grid(linestyle=':')
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('BinaryCrossentropy')
    plt.ylim([0, max(plt.ylim())])
    # plt.savefig("color_classification_model_for_mango_weight.png")
    plt.show()


if __name__ == '__main__':
    # weight level: >= 0.65 : 0, < 0.365 : 1
    train_path = 'dataset/100_100/weight/train/'
    train_label_path_0 = 'dataset/100_100/weight/train/0'
    train_label_path_1 = 'dataset/100_100/weight/train/1'

    test_path = 'dataset/100_100/weight/test/'
    test_label_path_0 = 'dataset/100_100/weight/test/0'
    test_label_path_1 = 'dataset/100_100/weight/test/1'

    train_img_0, train_label_0 = load_path(train_label_path_0, 0)
    train_img_1, train_label_1 = load_path(train_label_path_1, 1)

    test_img_0, test_label_0 = load_path(test_label_path_0, 0)
    test_img_1, test_label_1 = load_path(test_label_path_1, 1)

    # print(train_img_0)

    print("level 1 for weight train:", len(os.listdir(train_label_path_0)), "images")
    print("level 2 for weight train:", len(os.listdir(train_label_path_1)), "images")
    print('--' * 50)
    print("level 1 for weight new_data:", len(os.listdir(test_label_path_0)), "images")
    print("level 2 for weight new_data:", len(os.listdir(test_label_path_1)), "images")

    train_x = np.concatenate([train_img_0, train_img_1])
    train_y = np.concatenate([train_label_0, train_label_1])

    # print(train_x)
    # print(train_y.shape)

    train_x, train_y = shuffle(train_x, train_y)

    # print(train_x.shape)
    # print(train_y.shape)

    train_y = train_y.reshape(-1, 1)

    # print(train_y)

    # 將分類特徵編碼為onehot數值數組
    # train_y = onehot_encoder(train_y)

    # print(train_y)
    # print(test_y.shape)

    model = sequential()
    # model = vgg19()
    # model = color_classification()
    history = compile(model, train_x / 255., train_y)
    # model.save('color_classification_model_for_mango_weight.tf')
    training_model_plot(history)

    # predict
    y_true = np.concatenate([test_label_0, test_label_1])
    test_image = np.concatenate([test_img_0, test_img_1])

    # print(y_true.shape)
    # print(test_image)

    # model = load_model('color_classification_model_for_mango_weight.tf')
    # model.summary()

    y_pred = model.predict(test_image)
    # y_pred = np.argmax(y_pred, axis=1)

    print(y_pred)
    # print(y_pred.shape)

    score = accuracy_score(y_true, y_pred.astype('int'))
    print("weight accuracy:", score * 100, "%")
