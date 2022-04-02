import numpy as np
import cv2
from models import vgg16ModelFromScratch, vgg16ModelPretrained, designedCNNModel
import tensorflow as tf
from tensorflow import gather
import os

# define the file paths to the test images
file_path_test = './svhnData/test_images/'

# use the best-performing model to recognize digits in test image
def digits_detection_and_recognition(test_image):
    # duplicate input image
    image_copy = np.copy(test_image)
    # convert color image to grayscale
    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # get image width and height
    h, w = image_gray.shape

    # create mser object
    mser = cv2.MSER_create(_max_variation = 0.1)

    # detect ROIs in grayscale image
    rois, _ = mser.detectRegions(image_gray)
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in rois]
    # cv2.polylines(image_copy, hulls, 1, (0, 255, 0))

    # initialize a list to store all filtered ROIs
    rois_filtered = []
    # initialize a list to store all image patches in the filtered ROIs
    patches_filtered = []

    # loop through and filter the found ROIs
    for bbox in rois:
        # get upper left and lower right corners
        x_ul, y_ul = np.amin(bbox, axis=0)
        x_lr, y_lr = np.amax(bbox, axis=0)
        # get bbox width and height
        bbox_width = x_lr - x_ul
        bbox_height = y_lr - y_ul

        # filter out very small bboxes, very large bboxes, very short bboxes
        if bbox_width <= 0.05 * w or bbox_height <= 0.05 * h or \
                (bbox_width > 0.5 * w and bbox_height > 0.5 * h) or \
                bbox_width >= 1.5 * bbox_height or bbox_height >= 10 * bbox_width:
            continue
        else:
            # get the filtered image patch
            patch = image_gray[y_ul: y_lr, x_ul: x_lr]
            # add filtered image patch to patches
            patches_filtered.append(patch)
            # add corresponding bboxes to rois
            rois_filtered.append((x_ul, y_ul, x_lr, y_lr))
    # return original image if no rois detected
    if len(patches_filtered) == 0:
        return test_image
    # detect digits in the rois
    else:
        # initialize a list to store detected rois resized to 32x32
        patches_resized = []
        # initialize a list to store detected digits
        digits_detected = []
        # loop through detected rois
        for roi in patches_filtered:
            # resize rois
            roi_resized = cv2.resize(roi, (32, 32)).reshape((32, 32, 1))
            patches_resized.append(roi_resized)
        # print(np.array(patches_resized).shape)

        # predict digits in the resized roi
        labels_predicted = vgg16ModelPretrained.digit_prediction(
            "vgg16_pretrained_model.json", "vgg16_pretrained_weights.h5", np.array(patches_resized))
        # remove the non-digit label class '10'
        for index, detects in enumerate(labels_predicted):
            # get the maximum detected label
            detect_max = np.amax(detects)
            label = np.argmax(detects)
            if detect_max >= 0.5 and label != 10:
                digits_detected.append(label)
            else:
                digits_detected.append(None)

        # initialize input variables for the keras non_max_suppression function
        boxes = []
        scores = []
        # loop through the detected digits
        for index, digit in enumerate(digits_detected):
            if digit is not None:
                # add label to scores
                scores.append(digit)
                # add bbox to boxes
                boxes.append(list(rois_filtered[index]))
            else:
                continue
        # implement the nms algorithm
        selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.1)
        selected_boxes = gather(boxes, selected_indices)
        # print(selected_indices)
        # print(selected_boxes)

        # draw the boxes in test image with labels
        for i in range(len(selected_indices)):
            x1, y1, x2, y2 = selected_boxes[i]
            test_image = cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            test_image = cv2.putText(test_image, str(scores[selected_indices[i]]), org = (x1, y2 + 3),
                                     fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                     color = (0, 0, 255), thickness = 2, fontScale = 2)

    return test_image

# # test function
# # specify outputs folder
# outputs_folder = './graded_images'
# if not os.path.exists(outputs_folder):
#     os.makedirs(outputs_folder)
#
#
# for image_name in ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png"]:
#     image = cv2.imread(os.path.join(file_path_test, image_name))
#     output = digits_detection_and_recognition(image)
#     cv2.imwrite('./graded_images/{}'.format(image_name), output)
