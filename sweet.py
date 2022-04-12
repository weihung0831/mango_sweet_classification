import pretty_errors
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
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

    print(image.shape)

    return image, label


def onehot_encoder(label):
    onehot = OneHotEncoder()
    onehot.fit(label)
    onehot = onehot.transform(label).toarray()

    return onehot


def sequential():
    model = Sequential()
    model.add(Conv2D(input_shape=(100, 100, 4), filters=16, kernel_size=(3, 3), padding='same', activation='swish'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='swish'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='swish'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=2048, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    return model


def vgg16():
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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()

    # img = tf.keras.utils.plot_model(model, to_file='model/sweet_model/model.png', show_shapes=True)

    return model


def compile(model, x, y):
    model = tf.keras.models.Sequential()
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.95, staircase=False)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='BinaryCrossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=0, mode='max',
                                   baseline=None, restore_best_weights=True)
    history = model.fit(x, y, epochs=100, batch_size=2, validation_split=0.25, callbacks=[early_stopping])

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
    plt.savefig("./model/100_100/sweet_model/sequential_model_for_mango_sweet.png")
    plt.show()


if __name__ == '__main__':
    # sweetness level: > 14 : 0, < 14 : 1
    train_path = 'dataset/100_100/sweet/train/'
    train_label_path_0 = 'dataset/100_100/sweet/train/0'
    train_label_path_1 = 'dataset/100_100/sweet/train/1'

    test_path = 'dataset/100_100/sweet/test/'
    test_label_path_0 = 'dataset/100_100/sweet/test/0'
    test_label_path_1 = 'dataset/100_100/sweet/test/1'

    train_img_0, train_label_0 = load_path(train_label_path_0, 0)
    train_img_1, train_label_1 = load_path(train_label_path_1, 1)

    test_img_0, test_label_0 = load_path(test_label_path_0, 0)
    test_img_1, test_label_1 = load_path(test_label_path_1, 1)

    print("level 1 for sweetness train:", len(os.listdir(train_label_path_0)), "images")
    print("level 2 for sweetness train:", len(os.listdir(train_label_path_1)), "images")
    print('--' * 50)
    print("level 1 for sweetness new_data:", len(os.listdir(test_label_path_0)), "images")
    print("level 2 for sweetness new_data:", len(os.listdir(test_label_path_1)), "images")

    train_x = np.concatenate([train_img_0, train_img_1])
    train_y = np.concatenate([train_label_0, train_label_1])

    # print(train_x.shape)
    # print(train_y.shape)

    train_x, train_y = shuffle(train_x, train_y)

    # print(train_x.shape)
    # print(train_y.shape)

    train_y = train_y.reshape(-1, 1)

    # print(train_y)

    # 將分類特徵編碼為onehot數值數組
    # train_y = onehot_encoder(train_y)
    # test_y = onehot_encoder(test_y)

    # print(train_y)
    # print(test_y.shape)

    model = sequential()
    # model = vgg16()
    history = compile(model, train_x / 255., train_y)
    model.save('./model/100_100/sweet_model/sequential_model_for_mango_sweet.tf')
    training_model_plot(history)

    # predict
    y_true = np.concatenate([test_label_0, test_label_1])
    test_image = np.concatenate([test_img_0, test_img_1])

    # print(y_true.shape)
    # print(test_image.shape)

    # model = load_model('vgg16_model_for_mango_sweet.tf')
    # model.summary()

    y_pred = model.predict(test_image)
    # y_pred = np.argmax(y_pred, axis=1)

    print(y_pred)
    # print(y_pred.shape)

    score = accuracy_score(y_true, y_pred.astype('int'))
    print("sweet accuracy:", score * 100, "%")
