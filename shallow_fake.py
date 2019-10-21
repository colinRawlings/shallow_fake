#!python

################################################################################
# Imports
################################################################################

import os
import glob

from menpodetect import load_dlib_frontal_face_detector
import menpo.io as mio
from menpofit.aam.pretrained import load_balanced_frontal_face_fitter
from menpo.visualize import MatplotlibRenderer
from menpo.transform import ThinPlateSplines
from tqdm import tqdm

################################################################################
# Definitions
################################################################################

COLIN_IMG_FILEPATH = os.path.join(os.path.dirname(__file__), 'assets', 'Colin-cropped.jpg')
COSIMO_VID_FILEPATH = os.path.join(os.path.dirname(__file__), 'assets', 'cosimo-cropped.mov')
COSIMO_IMGS_DIRPATH = os.path.join(os.path.dirname(__file__), 'assets', 'cosimo_imgs')

FRAME_FILENAME_PATTERN = 'frame_{:04d}.png'
COSIMO_IMGS_FILEPATTERN = os.path.join(os.path.dirname(__file__), 'assets', 'cosimo_imgs', FRAME_FILENAME_PATTERN)
OUTPUT_DIRNAME = os.path.join(os.path.dirname(__file__), 'output')
OUTPUT_FILEPATTERN = os.path.join(OUTPUT_DIRNAME, FRAME_FILENAME_PATTERN)


################################################################################
# Functions
################################################################################

def convert_video_to_frames(video_filename, dest_folder=None,
                            image_format="png", start_time=None, duration=None):
    import os
    import subprocess as sp
    import glob

    frame_pattern = "frame_%04d"
    avconv_name = "avconv"

    #

    if dest_folder is None:
        import python_helpers as yh
        dest_folder = yh.get_tmp_folder_name()

    os.makedirs(dest_folder)
    file_pattern_video_frames = dest_folder + os.path.sep + frame_pattern + "." \
                                + image_format

    #

    cmd_list = [avconv_name, ]

    if start_time is not None:
        cmd_list.append('-ss')
        cmd_list.append(str(start_time))

    if duration is not None:
        cmd_list.append('-t')
        cmd_list.append(str(duration))

    cmd_list = cmd_list + ["-i", os.path.abspath(video_filename),
                           "-qscale", "1", file_pattern_video_frames]

    assert sp.call(cmd_list) == 0, "Conversion failed"

    #

    all_files = glob.glob(dest_folder + os.path.sep + "*." + image_format)
    number_of_frames = len(all_files)

    return file_pattern_video_frames, number_of_frames


################################################################################

def get_landmarks(image_filepath, do_plot=True):
    assert os.path.isfile(image_filepath), f"img not found: {image_filepath}"
    img = mio.import_image(image_filepath)

    #

    detector = load_dlib_frontal_face_detector()
    bbs = detector(img)
    assert len(bbs) == 1, f"Unexpected num bb: {len(bbs)}"

    #

    fitter = load_balanced_frontal_face_fitter()
    lm_result = fitter.fit_from_bb(img, bbs[0])

    #

    img.landmarks['ibug_0'] = lm_result.final_shape
    img.landmarks.pop('dlib_0')

    #

    mr = None

    if do_plot:
        mr = MatplotlibRenderer(None, True)
        img.view(mr.figure_id)
        bbs[0].view(mr.figure_id)
        lm_result.view(mr.figure_id)

    return img, lm_result, mr


################################################################################

def prep_source_img(src, crop=True, do_plot=True):
    src = src.as_masked()

    if crop:
        src = src.crop_to_landmarks()

    src = src.constrain_mask_to_landmarks()

    if do_plot:
        src.view(new_figure=True)

    return src


################################################################################

def warp_face_onto_background(face_to_warp_filepath, background_face_filepath, do_plot=True, do_all_plots=False):
    img_warp_face, lm_warp_face, mr_warp_face = get_landmarks(face_to_warp_filepath, do_plot=do_all_plots)
    img_bgd_face, lm_bgd_face, mr_bgd_face = get_landmarks(background_face_filepath, do_plot=do_all_plots)

    # prep warp_face

    img_warp_face_masked = prep_source_img(img_warp_face, do_plot=do_all_plots)
    img_bgd_face_masked = prep_source_img(img_bgd_face, crop=False, do_plot=do_all_plots)

    #

    tps_bgd_face_to_warp_face = ThinPlateSplines(img_bgd_face_masked.landmarks['ibug_0'].lms,
                                                 img_warp_face_masked.landmarks['ibug_0'].lms)
    warped_warp_face_to_bgd_face_tps = img_warp_face_masked.as_unmasked(copy=False).warp_to_mask(
        img_bgd_face_masked.mask, tps_bgd_face_to_warp_face)

    if do_plot:
        warped_warp_face_to_bgd_face_tps.view(new_figure=True)

    return warped_warp_face_to_bgd_face_tps


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # prep input

    if not os.path.isdir(COSIMO_IMGS_DIRPATH):
        file_pattern_video_frames, number_of_frames = convert_video_to_frames(COSIMO_VID_FILEPATH,
                                                                              COSIMO_IMGS_DIRPATH)

    num_images = len(glob.glob(os.path.join(COSIMO_IMGS_DIRPATH, 'frame_*.png')))

    # prep output

    os.makedirs(OUTPUT_DIRNAME, exist_ok=True)

    existing_output = glob.glob(os.path.join(OUTPUT_DIRNAME, '*.png'))
    print(f"Removing {len(existing_output)} files in in {OUTPUT_DIRNAME}")
    for path in existing_output:
        os.remove(path)

    # process

    print(f"Preparing to process {num_images} images")

    for idx in tqdm(range(num_images)):
        warped_face = warp_face_onto_background(COLIN_IMG_FILEPATH, COSIMO_IMGS_FILEPATTERN.format(idx + 1),
                                                do_plot=False)
        mio.export_image(warped_face, OUTPUT_FILEPATTERN.format(idx + 1))
