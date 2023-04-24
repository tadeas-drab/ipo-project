import cv2
import numpy as np
from datetime import datetime
import argparse
import os
import json

parser = argparse.ArgumentParser(description="No ML Img identification algorithm",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--identification_id", help="Identification id")
parser.add_argument("-r", "--reference_img", help="Reference img name")
parser.add_argument("-p", "--process_reference", action="store_true", help="Process reference or everyone")
args = parser.parse_args()
config = vars(args)
print(config)

identification_id = config.get("identification_id")
reference_img_name = config.get("reference_img")
process_reference = config.get("process_reference")

startTime = datetime.now().microsecond / 1000

directory_path = "./identification/" + identification_id + "/"
print(process_reference)

already_compared = {}
comparison_data = {}
combinations = 0


def with_reference_object(ref_img_name):
    global identification_id, directory_path, already_compared, combinations
    # Load the identification
    reference_image_raw = cv2.imread(directory_path + ref_img_name)

    # Convert the identification to gray scale
    reference_gray = cv2.cvtColor(reference_image_raw, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.1, nlevels=10, edgeThreshold=18)

    reference_keypoints, reference_descriptor = orb.detectAndCompute(reference_gray, None)

    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files
    for file in files:
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(directory_path, file)) and not file.lower().endswith(('.txt', '.json')) and file != ref_img_name \
               and (ref_img_name not in already_compared or file not in already_compared.get(ref_img_name)) \
                and (file not in already_compared or ref_img_name not in already_compared.get(file)):
            image = cv2.imread(os.path.join(directory_path, file))
            print("--------------")
            print(os.path.join(directory_path, file))

            test_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

            # Create a FLANN Matcher object
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=11,
                                key_size=19,
                                multi_probe_level=1)
            search_params = dict(checks=80)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Match the descriptors using FLANN
            matches = flann.knnMatch(reference_descriptor, test_descriptor, k=2)

            # Apply the ratio test to filter out poor matches
            good_matches = []

            for match in matches:
                if len(match) >= 2:
                    m, n = match
                    if m.distance < 0.785 * n.distance:
                        good_matches.append(m)

            if ref_img_name in already_compared:
                already_compared[ref_img_name].append(file)
            else:
                already_compared[ref_img_name] = [file]

            combinations = combinations + 1

            # Use RANSAC to estimate a homography between the images
            if len(good_matches) >= 4:
                src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
                matches_mask = mask.ravel().tolist()

                # Calculate the similarity percentage
                num_inliers = sum(matches_mask)
                similarity_percentage = (num_inliers * 100) / len(good_matches)

                print(f"Similarity Percentage: {similarity_percentage:.2f}%")

                if ref_img_name in comparison_data:
                    comparison_data[ref_img_name].append([file, similarity_percentage])
                else:
                    comparison_data[ref_img_name] = [[file, similarity_percentage]]
            else:
                print("Not enough good matches to calculate similarity")

                if ref_img_name in comparison_data:
                    comparison_data[ref_img_name].append([file, 0])
                else:
                    comparison_data[ref_img_name] = [[file, 0]]


def with_each_other():
    global already_compared
    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files
    for file in files:
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(directory_path, file)) and not file.lower().endswith(('.txt', '.json')):
            print(" ")
            print("REFERENCE:")
            print(file)
            with_reference_object(ref_img_name=file)
            print(already_compared[file] if file in already_compared else "NONE")
            print("Combinations: " + str(combinations))

if process_reference is True:
    with_reference_object(ref_img_name=reference_img_name)
else:
    with_each_other()

endTime = datetime.now().microsecond / 1000

print("Time spent: " + str(endTime - startTime))
print("Combinations: " + str(combinations))
# Serializing json
json_object = json.dumps(comparison_data, indent = 4)
print(json_object)

# Open the file for writing
with open(directory_path + "output-no-ml.json", "w") as file:
    # Write the content to the file
    file.write(json_object)
