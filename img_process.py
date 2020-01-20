from torchvision import transforms
import cv2
import numpy as np
from torchvision.transforms import functional as F


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes


class Resize(object):
    def __call__(self, image, boxes):
        sh = image.shape
        image = cv2.resize(image, (1024, 512))
        for i, axis in enumerate(boxes):
            if i % 2 == 0:
                boxes[i] = (axis / sh[1]) * 1024
            if i % 2 == 1:
                boxes[i] = (axis / sh[0]) * 512
        return image, boxes


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target


class CLAHE(object):
    def __call__(self, image, boxes=None):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image, boxes


class PreProcess(object):
    def __init__(self, mean=125):
        self.mean = mean
        self.augment = Compose([
            CLAHE(),
            Resize(),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes):
        return self.augment(img, boxes)
