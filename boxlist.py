import torch
from torchvision import ops


FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList:
    def __init__(self, box):
        device = box.device if hasattr(box, 'device') else 'cpu'
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        self.box = box

        self.fields = {}

    def __len__(self):
        return self.box.shape[0]

    def area(self):
        box = self.box
        remove = 1
        area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)
        return area

    def __getitem__(self, index):
        box = BoxList(self.box[index])

        for k, v in self.fields.items():
            box.fields[k] = v[index]

        return box

    def to(self, device):
        box = BoxList(self.box.to(device))

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def cat_boxlist(boxlists):
    field_keys = boxlists[0].fields.keys()

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxList(box_cat)

    for field in field_keys:
        data = torch.cat([boxlist.fields[field] for boxlist in boxlists], 0)
        new_boxlist.fields[field] = data

    return new_boxlist


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]

    return boxlist
