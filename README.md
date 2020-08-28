# Bird Detection
Detecting birds which feed on cash crops with object detection.
This turns regular MobilenetV2 models into an object detector.

# Usage

## Build Dataset
This method uses selective search to find image using the 
annotations and create a new dataset for training.

```bash
python build_dataset.py
```

## Training
Run fine_tune_rcnn.py to train the model. It will read the build 
dataset and saves the model and encoded labels.

```bash
python fine_tune_rcnn.py
```

## Running Inference
Run detect_bird_rcnn.py to detect bird in the image.

```bash
python detect_bird_rcnn.py -i images/target_1.jpg
```

# Reference
[pyimagesearch](https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
)
