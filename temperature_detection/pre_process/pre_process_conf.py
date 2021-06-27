from datetime import datetime


class PreProcessConf:
    log_dir = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    IMAGES_TO_SUMMARY = 8
    DISPLAY_IMAGES = True
