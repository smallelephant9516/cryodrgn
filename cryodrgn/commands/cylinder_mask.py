import numpy as np
import mrcfile
import cv2
import argparse
import os, sys
from datetime import datetime as dt
import matplotlib.pyplot as plt

start = dt.now()


def create_soft_circular_mask(h, w, center=None, radius=None, soft_start=0.9):
    # Set default center if none is provided
    if center is None:
        center = (int(w / 2), int(h / 2))

    # Set default radius if none is provided
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    radius = int(radius)

    # Calculate the hard and soft boundary distances from the center
    hard_radius = radius * soft_start
    soft_radius = radius

    # Create a 2D grid with indices for each point
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # Initialize mask with zeros
    mask = np.zeros((h, w))

    # Hard region: Full mask value inside 90% of the radius
    mask[dist_from_center <= hard_radius] = 1

    # Soft region: Gradually decrease from 1 to 0 between 90% and 100% of the radius
    soft_region = (dist_from_center > hard_radius) & (dist_from_center <= soft_radius)
    mask[soft_region] = 1 - (dist_from_center[soft_region] - hard_radius) / (soft_radius - hard_radius)

    return mask

def create_soft_horizental_mask(D, radius_width, soft_start=0.9):
    mask = np.zeros((D, D))

    center = D / 2

    # Calculate the hard and soft boundary distances from the center
    hard_radius = radius_width * soft_start
    soft_radius = radius_width

    # Create a 2D grid with indices for each point
    Y, X = np.ogrid[:D, :D]

    Y_repeat = np.repeat(Y, D, axis=1)

    dist_from_center = np.abs(Y_repeat - center)

    mask[dist_from_center <= hard_radius] = 1
    soft_region = (dist_from_center > hard_radius) & (dist_from_center <= soft_radius)
    mask[soft_region] = 1 - (dist_from_center[soft_region] - hard_radius) / (soft_radius - hard_radius)

    return mask




def apply_mask(image, angle, mask):
    # Compute the rotation matrix

    dim = image.shape[0]
    mean = image.mean()
    image = image-mean
    rotation_matrix = cv2.getRotationMatrix2D((dim / 2, dim / 2), angle, 1.0)

    # Apply the rotation
    mask = cv2.warpAffine(mask, rotation_matrix, (dim, dim))

    image = image * mask
    image = image+mean

    return image


def rotate_image(image, angle):
    # Compute the rotation matrix

    dim = image.shape[0]
    rotation_matrix = cv2.getRotationMatrix2D((dim / 2, dim / 2), angle, 1.0)

    # Apply the rotation
    image = cv2.warpAffine(image, rotation_matrix, (dim, dim))

    return image


def add_args(parser):
    parser.add_argument('mrcs', help='Input mrcs file' )
    parser.add_argument('--rotation', action='store_true', help='whether rotate the image horizental or not')
    parser.add_argument('--psi_prior', metavar='npy', type=os.path.abspath, help='the prior psi angle')
    parser.add_argument('--pixel_size', type=float, help='take the whole filament as neighbor')
    parser.add_argument('--width', type=float, help='the diameter of the filament')
    parser.add_argument('--o', help='output filament id')
    return parser

def main(args):

    data_path = args.mrcs
    data_mrc = mrcfile.open(data_path, permissive=True)

    pixel_size = args.pixel_size

    data = data_mrc.data

    N, dim, _ = data.shape

    radius = int((args.width/2)/pixel_size)

    mask = np.ones((dim,dim))

    cylinder_mask = create_soft_horizental_mask(dim, radius)
    circular_mask = create_soft_circular_mask(dim, dim)
    mask=mask*cylinder_mask*circular_mask

    psi_angle = np.load(args.psi_prior)


    all_image = []
    for i in range(len(psi_angle)):
        angle = psi_angle[i]
        image = data[i]
        image = apply_mask(image, angle, mask)
        if args.rotation is True:
            image = rotate_image(image,-angle)
        all_image.append(image)
        if i % 1000 == 0:
            print(i)
    all_image = np.array(all_image).astype(np.float32)


    plt.imshow(all_image[90])
    plt.show()

    mrcfile.write(args.o, all_image, overwrite=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())