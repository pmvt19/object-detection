print("hello")

from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-mobilenet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"], help="name of the object detection model")
# ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle", help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

DEVICE = torch.device("cpu")

CLASSES = {0: u'__background__',
1 : u'person',
2 : u'bicycle',
3 : u'car',
4 : u'motorcycle',
5 : u'airplane',
6 : u'bus',
7 : u'train',
8 : u'truck',
9 : u'boat',
10 : u'traffic light',
11 : u'fire hydrant',
12 : u'street sign',
13 : u'stop sign',
14 : u'parking meter',
15 : u'bench',
16 : u'bird',
17 : u'cat',
18 : u'dog',
19 : u'horse',
20 : u'sheep',
21 : u'cow',
22 : u'elephant',
23 : u'bear',
24 : u'zebra',
25 : u'giraffe',
26 : u'hat',
27 : u'backpack',
28 : u'umbrella',
29 : u'shoe',
30 : u'eye glasses',
31 : u'handbag',
32 : u'tie',
33 : u'suitcase',
34 : u'frisbee',
35 : u'skis',
36 : u'snowboard',
37 : u'sports ball',
38 : u'kite',
39 : u'baseball bat',
40 : u'baseball glove',
41 : u'skateboard',
42 : u'surfboard',
43 : u'tennis racket',
44 : u'bottle',
45 : u'plate',
46 : u'wine glass',
47 : u'cup',
48 : u'fork',
49 : u'knife',
50 : u'spoon',
51 : u'bowl',
52 : u'banana',
53 : u'apple',
54 : u'sandwich',
55 : u'orange',
56 : u'broccoli',
57 : u'carrot',
58 : u'hot dog',
59 : u'pizza',
60 : u'donut',
61 : u'cake',
62 : u'chair',
63 : u'couch',
64 : u'potted plant',
65 : u'bed',
66 : u'mirror',
67 : u'dining table',
68 : u'window',
69 : u'desk',
70 : u'toilet',
71 : u'door',
72 : u'tv',
73 : u'laptop',
74 : u'mouse',
75 : u'remote',
76 : u'keyboard',
77 : u'cell phone',
78 : u'microwave',
79 : u'oven',
80 : u'toaster',
81 : u'sink',
82 : u'refrigerator',
83 : u'blender',
84 : u'book',
85 : u'clock',
86 : u'vase',
87 : u'scissors',
88 : u'teddy bear',
89 : u'hair drier',
90 : u'toothbrush',
91 : u'hair brush'}

# CLASSES = {0: u'__background__',
#  1: u'person',
#  2: u'bicycle',
#  3: u'car',
#  4: u'motorcycle',
#  5: u'airplane',
#  6: u'bus',
#  7: u'train',
#  8: u'truck',
#  9: u'boat',
#  10: u'traffic light',
#  11: u'fire hydrant',
#  12: u'stop sign',
#  13: u'parking meter',
#  14: u'bench',
#  15: u'bird',
#  16: u'cat',
#  17: u'dog',
#  18: u'horse',
#  19: u'sheep',
#  20: u'cow',
#  21: u'elephant',
#  22: u'bear',
#  23: u'zebra',
#  24: u'giraffe',
#  25: u'backpack',
#  26: u'umbrella',
#  27: u'handbag',
#  28: u'tie',
#  29: u'suitcase',
#  30: u'frisbee',
#  31: u'skis',
#  32: u'snowboard',
#  33: u'sports ball',
#  34: u'kite',
#  35: u'baseball bat',
#  36: u'baseball glove',
#  37: u'skateboard',
#  38: u'surfboard',
#  39: u'tennis racket',
#  40: u'bottle',
#  41: u'wine glass',
#  42: u'cup',
#  43: u'fork',
#  44: u'knife',
#  45: u'spoon',
#  46: u'bowl',
#  47: u'banana',
#  48: u'apple',
#  49: u'sandwich',
#  50: u'orange',
#  51: u'broccoli',
#  52: u'carrot',
#  53: u'hot dog',
#  54: u'pizza',
#  55: u'donut',
#  56: u'cake',
#  57: u'chair',
#  58: u'couch',
#  59: u'potted plant',
#  60: u'bed',
#  61: u'dining table',
#  62: u'toilet',
#  63: u'tv',
#  64: u'laptop',
#  65: u'mouse',
#  66: u'remote',
#  67: u'keyboard',
#  68: u'cell phone',
#  69: u'microwave',
#  70: u'oven',
#  71: u'toaster',
#  72: u'sink',
#  73: u'refrigerator',
#  74: u'book',
#  75: u'clock',
#  76: u'vase',
#  77: u'scissors',
#  78: u'teddy bear',
#  79: u'hair drier',
#  80: u'toothbrush'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}
print("Before Download")
# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=91, pretrained_backbone=True).to(DEVICE)
print("After Download")
model.eval()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=r'C:\Users\prava\Downloads\Sample_Video.mov').start()
time.sleep(2.0)
fps = FPS().start()
print(type(vs), vs)
# loop over the frames from the video stream
frame_counter = 0
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame_counter += 1
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	orig = frame.copy()
	# convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1))
	# add a batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the frame to a floating point tensor
	frame = np.expand_dims(frame, axis=0)
	frame = frame / 255.0
	frame = torch.FloatTensor(frame)
	# send the input to the device and pass the it through the
	# network to get the detections and predictions
	frame = frame.to(DEVICE)

	detections = model(frame)[0]
	# temp_ind = int(detections["labels"][0])
	# print(CLASSES[temp_ind])

    # print(detections)

    # loop over the detections
	for i in range(0, len(detections["boxes"])):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections["scores"][i]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# detections, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections["labels"][i])
			
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")
			print(frame_counter, CLASSES[idx], startX, startY)
			# draw the bounding box and label on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(orig, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # show the output frame
	cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()