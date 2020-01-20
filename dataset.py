
import os.path as osp
import torch.utils.data as data
import cv2
import xml.etree.ElementTree as ETree

import torch

from boxlist import BoxList


tooth_lab = (  # always index 0
    'RU8', 'RU7', 'RU6', 'RU5', 'RU4', 'RU3', 'RU2', 'RU1',
    'LU8', 'LU7', 'LU6', 'LU5', 'LU4', 'LU3', 'LU2', 'LU1',
    'RD8', 'RD7', 'RD6', 'RD5', 'RD4', 'RD3', 'RD2', 'RD1',
    'LD8', 'LD7', 'LD6', 'LD5', 'LD4', 'LD3', 'LD2', 'LD1',
    )


class VOCAnnotTransFaster(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(tooth_lab, range(len(tooth_lab))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        labels = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.upper().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                bndbox.append(cur_pt)
            # for xywh
            # bndbox[2] = bndbox[2] - bndbox[0]
            # bndbox[3] = bndbox[3] - bndbox[1]
            label_idx = self.class_to_ind[name]
            labels.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        target = BoxList(res)
        classes = torch.tensor(labels)
        target.fields['labels'] = classes

        # 1画像に複数物体あるので、[物体数,[bndbox]]のリストを作成する
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class ToothDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 doctor='Yamamoto',
                 transform=None,
                 target_transform=None):

        self.root = root_path
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', f'Annotations({doctor})', '%s.xml')
        self._imgpath = osp.join('%s', 'TIFFImages', '%s.tif')
        self.ids = list()

        for line in open(osp.join(root_path, 'ImageSets', 'trainval.txt')):
            self.ids.append((root_path, line.strip()))

    def __getitem__(self, index):
        im, gt, name, size = self.pull_item(index)

        return im, gt, name

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ETree.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, 1)
        height, width, channel = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            img, boxes = self.transform(img, target.box)
            target.box = boxes
        # 画像の次元の順番をHWCからCHWに変更
        return torch.from_numpy(img).permute(2, 0, 1), target, img_id, (height, width)
        # return torch.from_numpy(img), target, height, width


def detection_collate(batch):
    targets = []
    imgs = []
    name = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        name.append(sample[2])
    return torch.stack(imgs, 0), targets, name
