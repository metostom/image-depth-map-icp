import argparse
import cv2
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--ndisparity', default=16, type=int)
    parser.add_argument('--blocksize', default=15, type=int)   
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    
    return args


def read_images_grayscale(image_paths):

    return [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths]


def compute_depth(image_1, image_2, ndisparity, blocksize):

    stereo = cv2.StereoBM_create(numDisparities=ndisparity, blockSize=blocksize)
    disparity = stereo.compute(image_1, image_2)

    return disparity


def save_depth_map(out_array, out_file):

    cv2.imwrite(out_file, out_array)


def plot_depth_map(depth_array):

    plt.imshow(depth_array)
    plt.colorbar()
    plt.show()


def main():

    args = parse_args()
    imgs = read_images_grayscale(args.images)
    disparity = compute_depth(imgs[0], imgs[1], args.ndisparity, args.blocksize)

    if args.outfile:
        save_depth_map(disparity, args.outfile)

    if args.plot:
        plot_depth_map(disparity)


if __name__ == '__main__':

    main()

    