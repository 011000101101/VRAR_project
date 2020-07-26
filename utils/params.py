"""
global params used by several parts of the system
"""

import os

"""
width and height of image samples in pixels
"""
SAMPLE_IMAGE_SIZE = 25

"""
project root path
"""
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')  # This is your Project Root

"""
names and paths of available fonts
"""
FONT_PATHS = [
    ("komorebi_gothic", "MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"),
    ("kosugi_maru", "Kosugi_Maru/KosugiMaru-Regular.ttf"),
    # ("chi_jyun", "chi_jyun/é┐é═éÔÅâ.ttf"),
    ("togalite-light", "togalite/togalite-light.otf"),
    ("togalite-regular", "togalite/togalite-regular.otf"),
    ("togalite-medium", "togalite/togalite-medium.otf"),
    ("togalite-bold", "togalite/togalite-bold.otf"),
    # ("togalite-heavy", "togalite/togalite-heavy.otf"),
    # ("togalite-black", "togalite/togalite-black.otf"),
    # ("chikara_dzuyoku", "chikara_dzuyoku/851CHIKARA-DZUYOKU_kanaA_004.ttf"),
    # ("nikkyou_sans", "NikkyouSans/NikkyouSans-B6aV.ttf"),
    ("xano_mincho", "Xano-Mincho/XANO-mincho-U32.ttf")
]
