import cv2
import os
from digits_detection_and_recognition import digits_detection_and_recognition


# specify outputs folder
outputs_folder = './graded_images'
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

# file path of test images
imgs_dir = './svhnData/test_images/'
# output to current directory
output_dir = "./"
# read in all test images (five good images + two bad images)
imgs_list = [f for f in os.listdir(imgs_dir)
             if f[0] != '.' and f.endswith('.png')]
imgs_list.sort()

# process and save outputs
for img in imgs_list:
    test_image = cv2.imread(os.path.join(imgs_dir, img))
    output_image = digits_detection_and_recognition(test_image)
    cv2.imwrite('./graded_images/{}'.format(img), output_image)
