from datetime import datetime


# TODO - use this as "ProjectConf" conf and relocate to higher project level
class PreProcessConf:
    log_dir = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    SOURCE_IMAGE_DIR = './outputs/'
    AUGMENT_DIR = './augmented/'

    NUM_LABELS = 3

    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    IMAGES_TO_SUMMARY = 3
    DISPLAY_IMAGES = True

    N_SPLITS = 5
