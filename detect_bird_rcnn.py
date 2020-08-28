import argparse
import pickle

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mute_tf_warnings import tf_mute_warning

from utils import config
from utils.nms import non_max_suppression

tf_mute_warning()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading model and label binarizer...")
model = tf.keras.models.load_model(config.MODEL_PATH)
lb = pickle.load(open(config.ENCODER_PATH, "rb"))

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

# run selective search on the image to generate bounding box proposal 
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []

for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS,
                     interpolation=cv2.INTER_CUBIC)

    roi = tf.keras.preprocessing.image.img_to_array(roi)
    roi = tf.keras.applications.mobilenet_v2.preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print(f"[INFO] proposal shape: {proposals.shape}")

print("[INFO] classifying proposals...")
proba = model.predict(proposals)

print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "target")[0]

boxes = boxes[idxs]
proba = proba[idxs][:, 1]

# print(proba)

idxs = np.where(proba >= config.MIN_PROBA)
boxes = boxes[idxs]
proba = proba[idxs]

clone = image.copy()

# print(boxes)
for (box, prob) in zip(boxes, proba):
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    # text = f"Target: {prob * 100:.2f\\\}%"
    text = "Target"
    cv2.putText(clone, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# # cv2.imshow("Before NMS", clone)
# plt.imshow(clone)
# plt.title("Before NMS")
# plt.show()
boxes = non_max_suppression(boxes, proba)

for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    # text = f"Target: {proba[3] * 100:.2f}%"
    text = "Target"
    cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# # cv2.imshow("After NMS", image)
# # cv2.waitKey(0)
# plt.imshow(image)
# plt.title("After NMS")
# plt.show()

fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(clone, label="Before NMS")
ax[1].imshow(image, label="After NMS")
ax[0].set_title("Before NMS")
ax[1].set_title("After NMS")
plt.show()

