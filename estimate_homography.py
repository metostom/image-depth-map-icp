import argparse
import cv2
import numpy as np
import pickle

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', default=None)
    parser.add_argument('--outpath',  default=None)
    parser.add_argument('--H', action='store_true')

    args = parser.parse_args()
    
    return args


def read_images(image_paths):

    return [cv2.imread(img) for img in image_paths]


def get_matches_homography(image_array_0, image_array_1):


    def detect_describe_keypoints(image_array):
        
        descriptor = cv2.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image_array, None)
        
        return (keypoints, features)
    

    def match_points(features1, features2):

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(features1, features2)

        return matches


    def estimate_homography(matches, keypoints_0, keypoints_1):

        query_pts = np.array([keypoints_0[m.queryIdx].pt for m in matches]).reshape(-1,1,2).astype(np.float32)
        train_pts = np.array([keypoints_1[m.trainIdx].pt for m in matches]).reshape(-1,1,2).astype(np.float32)
 
        H = cv2.findHomography(query_pts, train_pts, cv2.RANSAC)

        return H


    keypoints_0, features_0 = detect_describe_keypoints(image_array_0)
    keypoints_1, features_1 = detect_describe_keypoints(image_array_1)
    matches = match_points(features_0, features_1)    
    H = estimate_homography(matches, keypoints_0, keypoints_1)

    return H


def save_homography_images(image_1, image_2, H, out_file):

    out_d = {
        'images': [image_1, image_2],
        'H': H
    }

    with open(out_file, 'wb') as f:
        pickle.dump(out_d, f)


def main():

    args = parse_args()
    images = read_images(args.images)
    H = get_matches_homography(images[0], images[1])
    
    if args.outpath:
        save_homography_images(args.images[0], args.images[1], H, args.outpath)

    if args.H:
        print(H)


if __name__ == '__main__':
    main()