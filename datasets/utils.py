import albumentations as A


class ImgAug(object):

    def __init__(self) -> None:
        self.augmentor = A.Compose([
            A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.3, 0.1), contrast_limit=(-0.3, 0.1)),
            # A.Blur(p=0.1, blur_limit=(3, 9)),
            # A.MotionBlur(p=0.2, blur_limit=(3, 11)),
            # A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
            # A.HueSaturationValue(p=0.5, val_shift_limit=10)
        ], p=0.5)

    def __call__(self, x):
        return self.augmentor(image=x)['image']