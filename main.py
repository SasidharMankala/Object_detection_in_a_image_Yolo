import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image")
args = vars(ap.parse_args())
confind = 0.5
threshold= 0.3
labelsPath = os.path.sep.join([ "E:\Yoloproject\Yolo_img\yolo-coco\coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([ "E:\Yoloproject\Yolo_img\yolo-coco\yolov3.weights"])
configPath = os.path.sep.join(["E:\Yoloproject\Yolo_img\yolo-coco\yolov3-320.cfg"])
print("loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layerOutputs = net.forward(ln)

imgGry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


boxes = []
confidences = []
classIDs = []
count = 0




for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confind:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, confind, threshold)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # color = [int(c) for c in COLORS[classIDs[i]]]
        crop = image[y-5: y+h+5 , x-5: x+w+5]
        image_name1 = LABELS[classIDs[i]] + '_'+ str(count) + '.png'
        image_name2 = 'Detected_Objects.png'
        image_path1 = os.path.join('E:\Yoloproject\Yolo_img\images\Cropped_Images', image_name1 )
        # image_path2 = os.path.join('E:\Yoloproject\Yolo_img\images\Detected_Image',image_name2)
        cv2.imwrite(image_path1,crop)
        


        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow(LABELS[classIDs[i]]+ '_'+ str(count), crop)
        count +=1
        # rectangle = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        print(LABELS[classIDs[i]].upper(), 'with a confidence of : ',  round((confidences[i]*100),4),'%')

# cv2.imwrite(image_path2, image)
cv2.waitKey(0)


