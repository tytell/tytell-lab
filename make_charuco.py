import os
import cv2
import numpy as np
import argparse
import yaml

import matplotlib.pyplot as plt
import matplotlib

ARUCO_DICTS = {
    (4, 50): cv2.aruco.DICT_4X4_50,
    (5, 50): cv2.aruco.DICT_5X5_50,
    (6, 50): cv2.aruco.DICT_6X6_50,
    (7, 50): cv2.aruco.DICT_7X7_50,
    (4, 100): cv2.aruco.DICT_4X4_100,
    (5, 100): cv2.aruco.DICT_5X5_100,
    (6, 100): cv2.aruco.DICT_6X6_100,
    (7, 100): cv2.aruco.DICT_7X7_100,
    (4, 250): cv2.aruco.DICT_4X4_250,
    (5, 250): cv2.aruco.DICT_5X5_250,
    (6, 250): cv2.aruco.DICT_6X6_250,
    (7, 250): cv2.aruco.DICT_7X7_250,
    (4, 1000): cv2.aruco.DICT_4X4_1000,
    (5, 1000): cv2.aruco.DICT_5X5_1000,
    (6, 1000): cv2.aruco.DICT_6X6_1000,
    (7, 1000): cv2.aruco.DICT_7X7_1000
}

def main():
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='Calibrate based on images of Charuco boards')

    parser.add_argument('config', nargs="+")

    parser.add_argument('--base_path', default='.',
                        help='Base path for data files')

    parser.add_argument('-nx', '--nsquaresx', type=int, 
                        help='Number of grid squares horizontally',
                        default=6)
    parser.add_argument('-ny', '--nsquaresy', type=int, 
                        help='Number of grid squares vertically',
                        default=6)
    parser.add_argument('-sz', '--square_length', type=float,
                        help = 'Size of square in mm',
                        default = 24.33)
    parser.add_argument('-mlen', '--marker_length', type=float,
                        help='Size of the Aruco marker in mm',
                        default=17)
    parser.add_argument('-mbits', '--marker_bits', type=int,
                        help='Information bits in the markers',
                        default=5)
    parser.add_argument('-dict','--dict_size', type=int,
                        help='Number of markers in the dictionary',
                        default=50)

    parser.add_argument('--image_size', type=int,
                        help="Size of the Charuo image in pixels")
    parser.add_argument('--margin_pixels', type=int,
                        help="Size of the image margin in pixels")
    
    parser.add_argument('-o', '--output_file', 
                        help="Output image file with the Charuco image")

    args = parser.parse_args()

    if args.config is not None:
        for config1 in args.config:
            with open(config1, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        
        args = parser.parse_args()

    arcuo_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[(args.marker_bits, args.dict_size)])
    board = cv2.aruco.CharucoBoard((args.nsquaresx, args.nsquaresy),
                                   args.square_length, args.marker_length,
                                   arcuo_dict)
    
    size_ratio = args.nsquaresx / args.nsquaresy
    img_size = (args.image_size, int(size_ratio * args.image_size))
    img = cv2.aruco.CharucoBoard.generateImage(board, img_size, marginSize=args.margin_pixels)

    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(args.output_file, img)

if __name__ == "__main__":
    main()

