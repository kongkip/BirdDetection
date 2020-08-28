from utils import config
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCHS = 5
BS = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = tf.keras.preprocessing.image.load_img(
        imagePath, target_size=config.INPUT_DIMS
    )
    image = tf.keras.preprocessing.image.img_to_array(
        image
    )
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

# (trainX, testX, trainY,  testY) = train_test_split(
#     data, labels, test_size=0.20, stratify=labels, random_state=42
# )

aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    # fill_mode="neareast"
)

baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(
    weights="imagenet", include_top=False,
    input_tensor=tf.keras.Input(shape=(224, 224, 3)),
)

headModel = baseModel.input
headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = tf.keras.layers.Flatten()(headModel)
headModel = tf.keras.layers.Dense(128, activation="relu")(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel)
headModel = tf.keras.layers.Dense(2, activation="softmax")(headModel)

model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(lr=INIT_LR)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

print("[INFO] training head...")
H = model.fit(
    aug.flow(data, labels),
    steps_per_epoch=len(data) // BS,
    # validation_data=(testX, testY),
    # validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

print("[INFO] evaluating network...")
predIdxs = model.predict(data, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(labels.argmax(axis=1),
                            predIdxs,
                            target_names=lb.classes_))

print("[INFO] saving mask detector model...")
tf.keras.models.save_model(model, config.MODEL_PATH)

print("[INFO] saving label encoder...")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
