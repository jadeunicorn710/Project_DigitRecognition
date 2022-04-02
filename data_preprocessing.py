import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat


# define the file paths to the datasets in two formats
file_path_format1 = './svhnData/Format1/'
file_path_format2 = './svhnData/Format2/'


# load in the format 2 SVHN dataset
def load_dataset_format2():
    # load in the train set
    train_set = loadmat(os.path.join(file_path_format2, 'train_32x32.mat'))
    # get the images and labels
    X_train = np.transpose(train_set['X'], (3, 0, 1, 2))
    y_train = train_set['y'][:, 0]

    # load in the test set
    test_set = loadmat(os.path.join(file_path_format2, 'test_32x32.mat'))
    # get the images and labels
    X_test = np.transpose(test_set['X'], (3, 0, 1, 2))
    y_test = test_set['y'][:, 0]

    return (X_train, y_train, X_test, y_test)

# # test function
# X_train, y_train, X_test, y_test = load_dataset_format2()
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# change class label '10' to '0'
def label_processing():
    X_train, y_train, X_test, y_test = load_dataset_format2()
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    return (X_train, y_train, X_test, y_test)


# # test function
# X_train, y_train, X_test, y_test = load_dataset_format2()
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
#


# change color images to grayscale for CNN
def grayscale_image():
    # load in train and test datasets
    X_train, y_train, X_test, y_test = label_processing()

    # initialize arrays to store the grayscale images for train and test sets
    X_train_processed = []
    X_test_processed = []

    for train_image in X_train:
        X_train_processed.append(cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY))


    for test_image in X_test:
        X_test_processed.append(cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY))


    X_train_processed = np.array(X_train_processed)
    X_test_processed = np.array(X_test_processed)

    return (X_train_processed, y_train, X_test_processed, y_test)

# # test function
# X_train_gray, y_train, X_test_gray, y_test = grayscale_image()
# print(X_train_gray.shape, y_train.shape)
# print(X_test_gray.shape, y_test.shape)
#
# # ================================================================================
# # # save the test dataset to local drive (already saved, no need to rerun)
# # ================================================================================
# np.save('./svhnData/Format2/X_test.npy', X_test_gray.reshape((X_test_gray.shape[0], X_test_gray.shape[1], X_test_gray.shape[2], 1)))
# np.save('./svhnData/Format2/y_test.npy', y_test)


# define a function to extract the bbox information in format 1 digitStruct
def extract_bbox(index, digit_struct):
    # initialize a dictionary to store the extracted info
    bbox_info = {}
    # get the key-value pairs in current bbox
    bbox_keys = digit_struct['/digitStruct/bbox']
    pairs = bbox_keys[index].item()
    # get values for each attribute in current bbox
    for attr in ['top', 'left', 'height', 'width', 'label']:
        bbox_keys = digit_struct[pairs][attr]
        bbox_values = [digit_struct[bbox_keys[()][i].item()][()][0][0]
                  for i in range(len(bbox_keys))] if len(bbox_keys) > 1 else [bbox_keys[()][0][0]]
        # add the key-value pair into dictionary
        bbox_info[attr] = bbox_values

    return bbox_info


# add in non-digit samples using format 1 SVHN dataset to format 2 train dataset
def load_non_digit_samples():
    # load for the train digitStruct
    train_digit_struct = h5py.File(os.path.join(file_path_format1, 'train', 'digitStruct.mat'))
    # get keys for image names
    name_keys = train_digit_struct['/digitStruct/name']

    # initialize images and labels for the non-digit train set
    X_train_non_digit = []
    y_train_non_digit = []

    # loop though the digitStruct and extract relevant information
    for i in range(name_keys.shape[0]):
        # get image name
        image_name = ''.join([chr(char[0]) for char in train_digit_struct[name_keys[i][0]][()]])
        # get bbox info
        bbox_info = extract_bbox(i, train_digit_struct)
        # get the farthest extents of all bounding boxes
        bbox_top = int(np.min(bbox_info['top']))
        bbox_bottom = int(np.max([box_top + box_height for box_top, box_height in zip(bbox_info['top'], bbox_info['height'])]))
        bbox_left = int(np.min(bbox_info['left']))
        bbox_right = int(np.max([box_left + box_width for box_left, box_width in zip(bbox_info['left'], bbox_info['width'])]))

        # get the current image
        img = cv2.imread(os.path.join(os.path.join(file_path_format1, 'train'), image_name))

        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # define a dictionary to store the available margins around the bounding boxes
        margins = {}
        # add margins around
        margins['top'] = bbox_top
        margins['bottom'] = img.shape[0] - bbox_bottom
        margins['left'] = bbox_left
        margins['right'] = img.shape[1] - bbox_right

        # locate the largest available margin
        margin_max_loc = max(margins, key=margins.get)

        # get an image patch within the widest margin around the bounding boxes
        if margin_max_loc == 'top':
            img_patch = cv2.resize(img[0: bbox_top, :min(img.shape[1], 32)], (32, 32))
            # add image into the non-digit train set
            X_train_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit train set
            y_train_non_digit.append(10)

        elif margin_max_loc == 'bottom':
            img_patch = cv2.resize(img[bbox_bottom:, :min(img.shape[1], 32)], (32, 32))
            # add image into the non-digit train set
            X_train_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit train set
            y_train_non_digit.append(10)

        elif margin_max_loc == 'left':
            img_patch = cv2.resize(img[:min(img.shape[0], 32), :bbox_left], (32, 32))
            # add image into the non-digit train set
            X_train_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit train set
            y_train_non_digit.append(10)

        elif margin_max_loc == 'right':
            img_patch = cv2.resize(img[:min(img.shape[0], 32), bbox_right:], (32, 32))
            # add image into the non-digit train set
            X_train_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit train set
            y_train_non_digit.append(10)

        # output non-digit image patch to local drive
        # NOTE: already generated & saved in local drive to save runtime
        # cv2.imwrite('./svhnData/Format2/non_digit_train/' + str(i + 1) + '.png', img_patch.reshape((32, 32, 1)))

    X_train_non_digit = np.array(X_train_non_digit)
    y_train_non_digit = np.array(y_train_non_digit)

    # load for the test digitStruct
    test_digit_struct = h5py.File(os.path.join(file_path_format1, 'test', 'digitStruct.mat'))
    # get keys for image names
    name_keys = test_digit_struct['/digitStruct/name']

    # initialize images and labels for the non-digit test set
    X_test_non_digit = []
    y_test_non_digit = []

    # loop though the digitStruct and extract relevant information
    for i in range(name_keys.shape[0]):
        # get image name
        image_name = ''.join([chr(char[0]) for char in test_digit_struct[name_keys[i][0]][()]])
        # get bbox info
        bbox_info = extract_bbox(i, test_digit_struct)
        # get the farthest extents of all bounding boxes
        bbox_top = int(np.min(bbox_info['top']))
        bbox_bottom = int(np.max([box_top + box_height for box_top, box_height in zip(bbox_info['top'], bbox_info['height'])]))
        bbox_left = int(np.min(bbox_info['left']))
        bbox_right = int(np.max([box_left + box_width for box_left, box_width in zip(bbox_info['left'], bbox_info['width'])]))

        # get the current image
        img = cv2.imread(os.path.join(os.path.join(file_path_format1, 'test'), image_name))

        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # define a dictionary to store the available margins around the bounding boxes
        margins = {}
        # add margins around
        margins['top'] = bbox_top
        margins['bottom'] = img.shape[0] - bbox_bottom
        margins['left'] = bbox_left
        margins['right'] = img.shape[1] - bbox_right

        # locate the largest available margin
        margin_max_loc = max(margins, key=margins.get)

        # get an image patch within the widest margin around the bounding boxes
        if margin_max_loc == 'top':
            img_patch = cv2.resize(img[0: bbox_top, :min(img.shape[1], 32)], (32, 32))
            # add image into the non-digit test set
            X_test_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit test set
            y_test_non_digit.append(10)

        elif margin_max_loc == 'bottom':
            img_patch = cv2.resize(img[bbox_bottom:, :min(img.shape[1], 32)], (32, 32))
            # add image into the non-digit test set
            X_test_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit test set
            y_test_non_digit.append(10)

        elif margin_max_loc == 'left':
            img_patch = cv2.resize(img[:min(img.shape[0], 32), :bbox_left], (32, 32))
            # add image into the non-digit test set
            X_test_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit test set
            y_test_non_digit.append(10)

        elif margin_max_loc == 'right':
            img_patch = cv2.resize(img[:min(img.shape[0], 32), bbox_right:], (32, 32))
            # add image into the non-digit test set
            X_test_non_digit.append(img_patch.reshape((32, 32, 1)))
            # add image label '10' into the non-digit test set
            y_test_non_digit.append(10)

        # output non-digit image patch to local drive
        # NOTE: already generated & saved in local drive to save runtime
        # cv2.imwrite('./svhnData/Format2/non_digit_test/' + str(i + 1) + '.png', img_patch.reshape((32, 32, 1)))

    X_test_non_digit = np.array(X_test_non_digit)
    y_test_non_digit = np.array(y_test_non_digit)

    return (X_train_non_digit, y_train_non_digit, X_test_non_digit, y_test_non_digit)

# # ========================================================================================================================
# # output 33402 32x32 non-digit image samples to local drive, no need to rerun this since it takes around 10 minutes to run
# # ========================================================================================================================
# X_train_non_digit, y_train_non_digit, X_test_non_digit, y_test_non_digit = load_non_digit_samples()
# print(X_train_non_digit.shape, y_train_non_digit.shape, X_test_non_digit.shape, y_test_non_digit.shape)

# ================================================================================
# # save the non-digit datasets to local drive (already saved, no need to rerun)
# ================================================================================
# np.save('./svhnData/Format2/X_train_non_digit.npy', X_train_non_digit)
# np.save('./svhnData/Format2/y_train_non_digit.npy', y_train_non_digit)
# np.save('./svhnData/Format2/X_test_non_digit.npy', X_test_non_digit)
# np.save('./svhnData/Format2/y_test_non_digit.npy', y_test_non_digit)


# add the newly generated 33402 32x32 non-digit samples and label classes '10' into the format 2 train dataset
def add_non_digit_samples():
    # get the original format 2 dataset
    X_train, y_train, X_test, y_test = grayscale_image()
    # get the added non-digit dataset extracted from format 1 dataset
    # X_train_non_digit, y_train_non_digit = load_non_digit_samples()

    # load the saved non-digit train dataset
    X_train_non_digit = np.load('./svhnData/Format2/X_train_non_digit.npy')
    y_train_non_digit = np.load('./svhnData/Format2/y_train_non_digit.npy')

    # combine the oroginal format 2 train dataset with add non-digit dataset
    X_train_combined = np.concatenate((X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)), X_train_non_digit))
    y_train_combined = np.concatenate((y_train, y_train_non_digit))

    # load the saved non-digit test dataset
    X_test_non_digit = np.load('./svhnData/Format2/X_test_non_digit.npy')
    y_test_non_digit = np.load('./svhnData/Format2/y_test_non_digit.npy')

    # combine the oroginal format 2 test dataset with add non-digit dataset
    X_test_combined = np.concatenate((X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)), X_test_non_digit))
    y_test_combined = np.concatenate((y_test, y_test_non_digit))


    return (np.array(X_train_combined), np.array(y_train_combined), np.array(X_test_combined), np.array(y_test_combined))

# # test function
# X_train_combined, y_train_combined, X_test_combined, y_test_combined = add_non_digit_samples()
# print(X_train_combined.shape, y_train_combined.shape, X_test_combined.shape, y_test_combined.shape)

# ================================================================================
# # save the combined datasets to local drive (already saved, no need to rerun)
# ================================================================================
# np.save('./svhnData/Format2/X_train_combined.npy', X_train_combined)
# np.save('./svhnData/Format2/y_train_combined.npy', y_train_combined)
# np.save('./svhnData/Format2/X_test_combined.npy', X_test_combined)
# np.save('./svhnData/Format2/y_test_combined.npy', y_test_combined)
