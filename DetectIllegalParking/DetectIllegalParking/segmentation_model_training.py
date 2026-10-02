#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import Cityscapes
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

# Constants
BACKGROUND_CLASS = 0
LABEL_MAPPING = {7: 1, 8: 2, 9: 3}  # 1: road, 2: sidewalk, 3: parking
NUM_SEG_CLASSES = len(LABEL_MAPPING) + 1  # including background
NUM_DET_CLASSES = 1

# Placeholders for global losses
ce_loss = None
dice_loss = None

class CityscapesSegDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.cs = Cityscapes(root, split=split, mode='fine', target_type='semantic')
        self.transforms = transforms

    def __len__(self):
        return len(self.cs)

    def __getitem__(self, idx):
        img, mask = self.cs[idx]
        mask = np.array(mask, dtype=np.int64)
        new_mask = np.zeros_like(mask, dtype=np.int64) + BACKGROUND_CLASS
        for rid, cid in LABEL_MAPPING.items():
            new_mask[mask == rid] = cid
        if self.transforms:
            aug = self.transforms(image=np.array(img), mask=new_mask)
            img, new_mask = aug['image'], aug['mask']
        else:
            img = transforms.ToTensor()(img)
            new_mask = torch.as_tensor(new_mask)
        return img, new_mask.long()

def collate_fn(batch): return list(zip(*batch))

def get_seg_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomScale((0.5, 2.0)), A.RandomCrop(512, 1024), A.HorizontalFlip(),
            A.ColorJitter(0.3, 0.3, 0.3),
            A.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(512, 1024),
        A.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@torch.no_grad()
def evaluate_seg_iou(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_SEG_CLASSES, device=device)
    uni = torch.zeros(NUM_SEG_CLASSES, device=device)
    for imgs, masks in tqdm(loader, desc='Val Seg', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(1)
        for c in range(NUM_SEG_CLASSES):
            inter[c] += ((preds == c) & (masks == c)).sum()
            uni[c]   += ((preds == c) | (masks == c)).sum()
    iou = inter / (uni + 1e-6)
    # exclude background class from mean IoU
    mean_iou = iou[1:].mean().item()
    return mean_iou, iou[1:].cpu().tolist()

def train_seg(model, loader, opt, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc='Train Seg', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        with autocast():
            preds = model(imgs)
            loss = ce_loss(preds, masks) + dice_loss(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def save_ckpt(path, state):
    torch.save(state, path)
    print(f"Saved {path}")

def load_ckpt(path, model, opt, scaler=None):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    if scaler and 'scaler' in ck: scaler.load_state_dict(ck['scaler'])
    return ck.get('epoch', 0) + 1, ck.get('best', 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['seg'], required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    if args.task == 'seg':
        tr_ds = CityscapesSegDataset(args.root, 'train', get_seg_transforms(True))
        val_ds = CityscapesSegDataset(args.root, 'val', get_seg_transforms(False))

        # Oversample images containing parking
        has_parking = []
        for idx in range(len(tr_ds.cs)):
            _, raw_mask = tr_ds.cs[idx]
            raw_arr = np.array(raw_mask, dtype=np.int64)
            has_parking.append(np.any(raw_arr == 9))
        sample_weights = [5 if hp else 1 for hp in has_parking]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3', encoder_weights='imagenet',
            in_channels=3, classes=NUM_SEG_CLASSES
        ).to(device)

        # Losses with class weights to mitigate imbalance
        class_weights = torch.tensor([1.0, 1.0, 1.0, 4.0], device=device)
        global ce_loss, dice_loss
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        dice_loss = smp.losses.DiceLoss(mode='multiclass')

        # Freeze encoder for initial epochs
        freeze_epochs = 5
        for p in model.encoder.parameters(): p.requires_grad = False
        opt = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

        start_epoch, best_miou = 1, 0
        if args.resume and os.path.isfile(args.resume):
            start_epoch, best_miou = load_ckpt(args.resume, model, opt, scaler)

        for ep in range(start_epoch, args.epochs+1):
            # Unfreeze encoder after freeze_epochs
            if ep == freeze_epochs + 1:
                for p in model.encoder.parameters(): p.requires_grad = True
                opt = optim.AdamW(model.parameters(), lr=args.lr * 0.1)
                scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
                print(f"Unfroze encoder; new learning rate = {args.lr * 0.1}")

            loss = train_seg(model, tr_ld, opt, device, scaler)
            miou, cls_iou = evaluate_seg_iou(model, val_ld, device)
            print(f"Epoch {ep}/{args.epochs} | Loss {loss:.4f} | mIoU {miou:.4f} | IoUs {cls_iou}")

            ck = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': ep,
                'best': best_miou
            }
            save_ckpt(f"seg_ep{ep}.pth", ck)
            if miou > best_miou:
                best_miou = miou
                save_ckpt('seg_best.pth', ck)

            scheduler.step()

        # Quantize and export
        model.cpu()
        qm = torch.quantization.quantize_dynamic(model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
        torch.jit.save(torch.jit.script(qm), 'seg_cpu.pt')

if __name__ == '__main__':
    main()
