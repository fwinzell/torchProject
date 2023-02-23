import cv2
import os
import numpy as np
import datetime
import math
import argparse
from scipy import ndimage
from skimage.io import imread


def rotate(image, angle, use_binary=False):
    dim = math.ceil(2 * math.sqrt(image.shape[0] ** 2 / 2))
    exp = dim - image.shape[0]

    exp_img = np.zeros(shape=(dim, dim, image.shape[2]), dtype=image.dtype)
    s = int(exp / 2)
    # s has to be even
    s = s + (s % 2)
    exp_img[s:s + image.shape[0], s:s + image.shape[1], :] = image

    # Mirroring
    top = image[0:s, :, :]
    top = top[::-1, :, :]
    exp_img[0:s, s:s + image.shape[1], :] = top

    bot = image[image.shape[0] - s:, :, :]
    bot = bot[::-1, :, :]
    exp_img[image.shape[0] + s:, s:s + image.shape[1], :] = bot

    left = image[:, 0:s, :]
    left = left[:, ::-1, :]
    exp_img[s:s + image.shape[0], 0:s, :] = left

    right = image[:, image.shape[1] - s:, :]
    right = right[:, ::-1, :]
    exp_img[s:s + image.shape[0], image.shape[1] + s:, :] = right

    if use_binary:
        exp_img = np.uint8(exp_img / 255)
    rotated = ndimage.rotate(exp_img, angle, reshape=False)
    # rotated = cv2_rotate_image(exp_img,angle)

    if use_binary:
        cropped = rotated[s:s + image.shape[0], s:s + image.shape[1], :] * 255
    else:
        cropped = rotated[s:s + image.shape[0], s:s + image.shape[1], :]

    return cropped


def rotate_and_save(patch, count, patch_save_dir):
    pi = 0
    while pi < 359:
        # rotation angle in degrees
        rotated = rotate(patch, pi)
        patch_path = os.path.join(patch_save_dir, 'patch_{}.png'.format(count))
        cv2.imwrite(patch_path, rotated)
        count += 1
        pi += 45

    return count


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir,
                  patch_size, step_size=256,
                  save_mask=True,
                  use_otsu=False,
                  ):
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]

    def _filter_contours(contours, hierarchy, min_area=255 * 255, max_n_holes=8):
        """
            Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            # print(a)
            # self.displayContours(img.copy(), cont)
            if a == 0: continue
            if min_area < a:
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[:max_n_holes]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > min_area:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    for i in range(len(slides)):
        full_path = os.path.join(source, slides[i])
        wsi = cv2.imread(full_path)

        img_hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, 8, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
        # self.displayContours(img.copy(), contours)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        foreground_contours, hole_contours = _filter_contours(contours,
                                                              hierarchy, )  # Necessary for filtering out artifacts

        if save_mask:
            mask = visWSI(wsi, foreground_contours, hole_contours)
            mask_path = os.path.join(mask_save_dir, slides[i])
            cv2.imwrite(mask_path, mask)
            # mask.save(mask_path)

        patch_slide_folder = os.path.join(patch_save_dir, os.path.splitext(slides[i])[0])
        os.makedirs(patch_slide_folder, exist_ok=True)
        count = 0
        for contour in foreground_contours:
            count = _patching(wsi, contour, count, patch_save_dir=patch_slide_folder, patch_size=patch_size)


def _patching(wsi, contour, count, patch_save_dir, patch_size):
    start_x, start_y, w, h = cv2.boundingRect(contour)
    img_h, img_w = wsi.shape[:2]
    stop_y = min(start_y + h, img_h - patch_size + 1)
    stop_x = min(start_x + w, img_w - patch_size + 1)

    step_size_x = math.floor((stop_x - start_x) / math.ceil((stop_x - start_x) / patch_size))
    step_size_y = math.floor((stop_y - start_y) / math.ceil((stop_y - start_y) / patch_size))

    x_range = np.arange(start_x, stop_x, step=step_size_x)
    y_range = np.arange(start_y, stop_y, step=step_size_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coords = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

    for pt in coords:
        if cv2.pointPolygonTest(contour, tuple(np.array(pt).astype(float)), measureDist=False) > -1:  # check that point is within contour
            patch = wsi[pt[1]:pt[1] + patch_size, pt[0]:pt[0] + patch_size, :]
            #count = rotate_and_save(patch, count, patch_save_dir)
            patch_path = os.path.join(patch_save_dir, 'patch_{}.png'.format(count))
            cv2.imwrite(patch_path, patch)
            count += 1

    return count


def visWSI(wsi, contours, holes_tissue, color=(0, 255, 0), hole_color=(0, 0, 255),
           line_thickness=2, scale=0.1):
    dim = (int(wsi.shape[1] * scale), int(wsi.shape[0] * scale))
    downsample = cv2.resize(wsi, dim)
    cv2.drawContours(downsample, scaleContourDim(contours, scale),
                     -1, color, line_thickness, lineType=cv2.LINE_8)

    for holes in holes_tissue:
        cv2.drawContours(downsample, scaleContourDim(holes, scale),
                         -1, hole_color, line_thickness, lineType=cv2.LINE_8)

    return downsample


def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str,
                    help='path to folder containing raw wsi image files',
                    default="/home/fi5666wi/Documents/Prostate_images/2022-04-20")
parser.add_argument('--step_size', type=int, default=150,
                    help='step_size')
parser.add_argument('--patch_size', type=int, default=149,
                    help='patch_size')
parser.add_argument('--save_dir', type=str,
                    help='directory to save processed data',
                    default="/home/fi5666wi/Documents/Prostate_images/Patches_149/2022-04-20")

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   }

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    # seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
    #              'keep_ids': 'none', 'exclude_ids': 'none'}
    # filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 8}
    # vis_params = {'vis_level': -1, 'line_thickness': 10}
    # patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    seg_and_patch(source=args.source,
                  save_dir=args.save_dir,
                  patch_save_dir=patch_save_dir,
                  mask_save_dir=mask_save_dir,
                  patch_size=args.patch_size)
