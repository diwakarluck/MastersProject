import cv2
import numpy as np

# image_path = os.path.join(os.getcwd(), "images")
# print(image_path)
image_to_detect = cv2.imread('images/scene11.jpg')
# print(image_to_detect.shape)
img_height = image_to_detect.shape[0]
img_width = image_to_detect.shape[1]

img_blob = cv2.dnn.blobFromImage(image_to_detect, 0.003922, (416,416), swapRB=True, crop=False)
# class_labels = ["person"]
# set of 80 class labels
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

#convert that to a numpy array to apply color mask to the image numpy array
class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors,(16,1))

yolo_models = cv2.dnn.readNetFromDarknet('datasets/yolov3.cfg','datasets/yolov3.weights')

yolo_layers = yolo_models.getLayerNames()
yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_models.getUnconnectedOutLayers()]

yolo_models.setInput(img_blob)
obj_detection_layers = yolo_models.forward(yolo_output_layer)

############## NMS Change 1 ###############
# initialization for non-max suppression (NMS)
# declare list for [class id], [box center, width & height[], [confidences]
class_ids_list = []
boxes_list = []
confidences_list = []
############## NMS Change 1 END ###########

for obj_detection_layer in obj_detection_layers:
    for obj_detection in obj_detection_layer:
        all_scores = obj_detection[5:]
        predicted_class_id = np.argmax(all_scores)
        prediction_confidence = all_scores[predicted_class_id]
        if prediction_confidence > 0.20:
            # get the predicted label
            predicted_class_label = class_labels[predicted_class_id]
            if(predicted_class_label == "person"):
                # obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = obj_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height

                    ############## NMS Change 2 ###############
                    # save class id, start x, y, width & height, confidences in a list for nms processing
                    # make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                ############## NMS Change 2 END ###########

############## NMS Change 3 ###############
# Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes
# Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

# loop through the final set of detections remaining after NMS and draw bounding box and write text
c = 0
for max_valueid in max_value_ids:
    max_class_id = max_valueid[0]
    box = boxes_list[max_class_id]
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]

    # get the predicted class id and label
    predicted_class_id = class_ids_list[max_class_id]
    predicted_class_label = class_labels[predicted_class_id]
    prediction_confidence = confidences_list[max_class_id]
    ############## NMS Change 3 END ###########

    end_x_pt = start_x_pt + box_width
    end_y_pt = start_y_pt + box_height

    # get a random mask color from the numpy array of colors
    box_color = class_colors[predicted_class_id]

    # convert the color numpy array as a list and apply to text and box
    box_color = [int(c) for c in box_color]

    # print the prediction in console
    predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
    print("predicted object {}".format(predicted_class_label))
    c = c + 1

    # draw rectangle and text in the image
    cv2.rectangle(image_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
    cv2.putText(image_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            #     # get a random mask color from the numpy array of colors
            #     box_color = class_colors[predicted_class_id]
            #
            #     # convert the color numpy array as a list and apply to text and box
            #     box_color = [int(c) for c in box_color]
            #
            #     # print the prediction in console
            #     predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            #     print("predicted object {}".format(predicted_class_label))
            #
            #     # draw rectangle and text in the image
            #     cv2.rectangle(image_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            #     cv2.putText(image_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, box_color, 1)
            # else:
            #     pass
            #     # bounding_box = obj_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            #     # (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
            #     # start_x_pt = int(box_center_x_pt - (box_width / 2))
            #     # start_y_pt = int(box_center_y_pt - (box_height / 2))
            #     # end_x_pt = start_x_pt + box_width
            #     # end_y_pt = start_y_pt + box_height
            #     #
            #     # # get a random mask color from the numpy array of colors
            #     # box_color = class_colors[predicted_class_id]
            #     #
            #     # # convert the color numpy array as a list and apply to text and box
            #     # box_color = [int(c) for c in box_color]
            #     #
            #     # # print the prediction in console
            #     # predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            #     # print("predicted object {}".format(predicted_class_label))
            #     #
            #     # # draw rectangle and text in the image
            #     # cv2.rectangle(image_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            #     # cv2.putText(image_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5),
            #     #             cv2.FONT_HERSHEY_SIMPLEX,
            #     #             0.5, box_color, 1)

# Write some Text

# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10,500)
# fontScale              = 1
# fontColor              = (255,255,255)
# lineType               = 2
#
# cv2.putText(image_to_detect,'Total number of humans: '+str(c),
#     bottomLeftCornerOfText,
#     font,
#     fontScale,
#     fontColor,
#     lineType)
# cv2.putText(image_to_detect, 'Total number of humans: '+str(c), (10,500),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
# print("Total humans present in the place:", c)
print("Human Detection using YOLOV3: Total humans present -> "+str(c))
cv2.imshow("Human Detection: Total humans present -> "+str(c), image_to_detect)
k = cv2.waitKey(0)
if(k == 27):
    cv2.destroyAllWindows()