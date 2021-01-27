import cv2
from albumentations import (
    Compose, HorizontalFlip, Blur,
    ToFloat, ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, RandomCrop, HueSaturationValue
)

AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.3),

        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        Blur(blur_limit=4, p=0.1),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        GaussianBlur(blur_limit=(5, 9), sigma_limit=3, always_apply=False, p=0.7),
        RandomCrop(350, 426, always_apply=False, p=0.2),
        HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=70, always_apply=False, p=0.5, ),
        ToFloat(max_value=255)
    ]
)
AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])
