#!python

################################################################################
# Imports
################################################################################

import os
import glob
import typing as ty
import subprocess as sp

import matplotlib.pyplot as plt

from menpodetect import load_dlib_frontal_face_detector
import menpo.io as mio
from menpofit.aam.pretrained import load_balanced_frontal_face_fitter
from menpo.transform import ThinPlateSplines
from tqdm import tqdm
from menpo.image import Image

import numpy as np

################################################################################
# Definitions
################################################################################

FRAME_FILENAME_PATTERN = 'frame_{:04d}.png'

OUTPUT_DIRNAME = os.path.join(os.path.dirname(__file__), 'output')
OUTPUT_FILEPATTERN = os.path.join(OUTPUT_DIRNAME, FRAME_FILENAME_PATTERN)

################################################################################
# Functions
################################################################################


def convert_video_to_frames(video_filename: str,
                            dest_folder: str,
                            image_format: str = "png",
                            start_time: ty.Optional[float] = None,
                            duration: ty.Optional[float] = None):

    _FRAME_PATTERN = "frame_%04d"
    _AVCONV_NAME = "ffmpeg"

    #

    os.makedirs(dest_folder)
    file_pattern_video_frames = dest_folder + os.path.sep + _FRAME_PATTERN + "." \
                                + image_format

    #

    cmd_list = [
        _AVCONV_NAME,
    ]

    if start_time is not None:
        cmd_list.append('-ss')
        cmd_list.append(str(start_time))

    if duration is not None:
        cmd_list.append('-t')
        cmd_list.append(str(duration))

    cmd_list = cmd_list + ["-i", os.path.abspath(video_filename), "-qscale", "1", file_pattern_video_frames]

    assert sp.call(cmd_list) == 0, "Conversion failed"

    #

    all_files = glob.glob(dest_folder + os.path.sep + "*." + image_format)
    number_of_frames = len(all_files)

    return file_pattern_video_frames, number_of_frames


################################################################################


def get_landmarks(image_filepath: str, bb_index: int = 0, do_plot: bool = True) -> Image:
    assert os.path.isfile(image_filepath), f"img not found: {image_filepath}"
    img = mio.import_image(image_filepath)

    #

    detector = load_dlib_frontal_face_detector()
    bbs = detector(img)

    assert bb_index <= len(bbs) - 1, f"Too few bbs found to satisfy bb_index: {bb_index}"

    # order bbs based on x position

    x_bb = list()
    for bb in bbs:
        x_bb.append(np.mean(bb.points[:, 0]))

    x_bb_index = np.argsort(x_bb)

    #

    fitter = load_balanced_frontal_face_fitter()
    lm_result = fitter.fit_from_bb(img, bbs[x_bb_index[bb_index]])

    tags = [tag for tag in img.landmarks]
    for tag in tags:
        img.landmarks.pop(tag)

    img.landmarks['ibug_0'] = lm_result.final_shape

    #

    plt.figure()
    if do_plot:
        img.view()
        bbs[bb_index].view()
        lm_result.view()

    return img


################################################################################


def prep_source_img(src: Image, crop: bool = True, do_plot: bool = True) -> Image:
    src = src.as_masked()

    if crop:
        src = src.crop_to_landmarks()

    src = src.constrain_mask_to_landmarks()

    if do_plot:
        src.view(new_figure=True)

    return src


################################################################################


def warp_face_onto_background(face_to_warp_filepath: str,
                              background_face_filepath: str,
                              do_plot: bool = True,
                              do_all_plots: bool = False) -> Image:
    img_warp_face = get_landmarks(face_to_warp_filepath, do_plot=do_all_plots)
    img_bgd_face = get_landmarks(background_face_filepath, do_plot=do_all_plots)

    # prep warp_face

    img_warp_face_masked = prep_source_img(img_warp_face, do_plot=do_all_plots)
    img_bgd_face_masked = prep_source_img(img_bgd_face, crop=False, do_plot=do_all_plots)

    #

    tps_bgd_face_to_warp_face = ThinPlateSplines(img_bgd_face_masked.landmarks['ibug_0'],
                                                 img_warp_face_masked.landmarks['ibug_0'])
    warped_warp_face_to_bgd_face_tps = img_warp_face_masked.as_unmasked(copy=False).warp_to_mask(
        img_bgd_face_masked.mask, tps_bgd_face_to_warp_face)

    if do_plot:
        warped_warp_face_to_bgd_face_tps.view(new_figure=True)

    return warped_warp_face_to_bgd_face_tps
