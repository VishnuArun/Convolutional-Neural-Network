'''
Run with 

python3 train_smallervgg.py --dataset dataset --model catDogCactus.model --labelbin lb.pickle

The dataset should be organized into folders, where the folder title is the category name.
E.g. my folder 'dataset' includes folders (full of the corresponding images)
'Cat', 'Dog', and 'Cactus'.  So I call my model catDogCactus and also 
pickle the label vector.

If you're running on your own system, you probably have a few items to install.
Pip should be able to help you out with keras, sklearn, imutils, opencv-python, tflearn, tensorflow;
e.g.

pip3 install keras

python3 train_smallervgg.py --dataset dataset --model FordPorscheLexus.model --labelbin lb.pickle

'''

# This is from an amazing blogger at PyImageSearch, Adrian Rosebrock:
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# Read his blog, it's amazing.

print(__doc__)

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

def resize_square(im_pth, desired_size = 96):
  im = cv2.imread(im_pth)
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))

  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)

  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)
  return new_im

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
  help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
  help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
  help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 80
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
  # load the image, pre-process it, and store it in the data list
  new_im = resize_square(imagePath, desired_size = IMAGE_DIMS[0])
  image = img_to_array(new_im)
  data.append(image)
 
  # extract the class label from the image path and update the
  # labels list
  label = imagePath.split(os.path.sep)[-2]
  labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
  data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
  labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
  horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
  depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
  metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
  aug.flow(trainX, trainY, batch_size=BS),
  validation_data=(testX, testY),
  steps_per_epoch=len(trainX) // BS,
  epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
