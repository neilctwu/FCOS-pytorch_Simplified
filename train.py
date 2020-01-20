import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from dataset import ToothDataset, detection_collate, VOCAnnotTransFaster
from img_process import PreProcess

from argument import get_args
from backbone import vovnet57
from model import FCOS


def train(epoch, loader, model, optimizer, device):
    model.train()

    pbar = tqdm(loader, dynamic_ncols=True)

    for images, targets, _ in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        pre_model = model.state_dict()
        pre_optim = optimizer.state_dict()

        _, loss_dict = model(images, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()

        loss = loss_cls + loss_box + loss_center

        if torch.isnan(loss):
            print('nan happened')

            model.load_state_dict(pre_model)
            optimizer.load_state_dict(pre_optim)

            del images, targets
            torch.cuda.empty_cache()
            continue
        else:
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_cls = loss_dict['loss_cls'].mean().item()
        loss_box = loss_dict['loss_box'].mean().item()
        loss_center = loss_dict['loss_center'].mean().item()

        pbar.set_description(
            (
                f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                f'box: {loss_box:.4f}; center: {loss_center:.4f}'
            )
        )


def data_sampler(dataset, shuffle):
    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    args = get_args()

    device = 'cuda'
    ROOT = 'D:/Dental_Panorama/RawData/LabelingDatasets'

    backbone = vovnet57(pretrained=False)
    model = FCOS(args, backbone)
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
    )

    dataset = ToothDataset(root_path=ROOT,
                           doctor='Yamamoto',
                           target_transform=VOCAnnotTransFaster(),
                           transform=PreProcess())

    train_loader = DataLoader(dataset,
                              args.batch,
                              num_workers=0,
                              shuffle=True,
                              collate_fn=detection_collate,
                              pin_memory=True)

    if args.checkpoint:
        chkpnt = torch.load(args.checkpoint)
        model.load_state_dict(chkpnt['model'])
        optimizer.load_state_dict(chkpnt['optim'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0000000005

    for epoch in range(args.epoch):
        train(epoch, train_loader, model, optimizer, device)

        # scheduler.step()

        if epoch % 50 == 0:
            torch.save(
                {'model': model.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/epoch-{epoch + 1}.pt',
            )
