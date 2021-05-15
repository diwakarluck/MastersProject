# step 1: import the required modules..
import numpy as np
import cv2

# step 2: Load preprocess and convert image to blob..

# load the image to detect, get width, height..
# resize to match input size, convert to blob to pass into model..

image_to_detect = cv2.imread('images/scene10.jpg')
# print(image_to_detect.shape)
img_height = image_to_detect.shape[0]
img_width = image_to_detect.shape[1]

img_blob = cv2.dnn.blobFromImage(image_to_detect, swapRB=True, crop=False)
# recommended scale factor is 0.007843, width, height of blob is 300, 300, mean of 255 is 127.5
# blob is binary large data..

# step 3: set class labels..
# set of 21 class labels in alphabetical order (background + rest of 20 classes)
class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
# step 4: load pre-trained model and get prediction..

# loading pretrained model from prototext and caffemodel files..
# input preprocessed blob into model and pass through the model..
# obtain the detection predcitions by the model using forward() method..

mobilenetssd = cv2.dnn.readNetFromTensorflow('datasets/maskrcnn_buffermodel.pb', 'datasets/maskrcnn_bufferconfig.txt')
mobilenetssd.setInput(img_blob)
(obj_detections_boxes, obj_detections_masks) = mobilenetssd.forward(["detection_out_final", "detection_masks"])
# returned obj_detections[0, 0, index, 1], 1 => will have the prediction class
# 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates..

# step 5: Loop over detections, get class label, box coordinates..

# loop over the detections..
no_of_detections = obj_detections_boxes.shape[2]
# print(no_of_detections)
c = 0
for index in np.arange(0, no_of_detections):
    prediction_confidence = obj_detections_boxes[0, 0, index, 2]
    # take only predictions with confidence more than 20%
    if prediction_confidence > 0.20:
        # get the predicted label..
        predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
        predicted_class_label = class_labels[predicted_class_index]
        if (predicted_class_label == "person"):
            c = c + 1
            # obtain the bounding box co-ordinates for actual image from resized image size..
            bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")

            # step 7: Draw rectangle and text, display the image..

            # print the prediction in console..
            predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
            print("predicted object {}: {}".format(index+1, predicted_class_label))
            # draw rectangle and text in the image..
            cv2.rectangle(image_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0, 255, 0), 2)
            cv2.putText(image_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# show the output image..
print("Human Detection using RCNN: Total humans present -> "+str(c))
cv2.imshow("Human Detection: Total humans present -> "+str(c), image_to_detect)

k = cv2.waitKey(0)
if(k == 27):
    cv2.destroyAllWindows()




















