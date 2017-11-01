import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import utils
from unet11 import Loss, UNet11

Size = Tuple[int, int]


class StreetDataset(Dataset):
    def __init__(self, root: Path, size: Size, limit=None, augmentation=False):
        self.image_paths = sorted(root.joinpath('images').glob('*.jpg'))
        self.mask_paths = sorted(root.joinpath('instances').glob('*.png'))
        self.augmentation = augmentation
        if limit:
            self.image_paths = self.image_paths[:limit]
            self.mask_paths = self.mask_paths[:limit]
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx], size=self.size)
        mask = load_mask(self.mask_paths[idx], size=self.size)

        if self.augmentation:
            img, mask = augment(img, mask)

        return utils.img_transform(img), torch.from_numpy(np.expand_dims(mask, 0))


def load_image(path: Path, size: Tuple, with_size: bool = False):
    image = utils.load_image(path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    if with_size:
        size = image.shape[:2]
        return image, size
    else:
        return image


def load_mask(path: Path, size: Tuple):
    class_of_interest = 13  # construction--flat--road
    mask = (cv2.imread(str(path), 0) == class_of_interest).astype(np.uint8)

    mask = (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.float32)

    return mask


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []

    dice = []

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        dice += [get_dice(targets, (outputs > 0.5).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float

    valid_dice = np.mean(dice)

    print('Valid loss: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_dice))
    metrics = {'valid_loss': valid_loss, 'dice_loss': valid_dice}
    return metrics


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()


def rotate(img, mask, angle):
    rows, cols, _ = img.shape

    M = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)), angle, 1)
    dst_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
    dst_mask = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
    return dst_img, dst_mask


def augment(img, mask, max_angle=10):
    if np.random.random() < 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random() < 0.5:  # rotations up to max_angle in both directions
        random_angle = (random.random() * 2 - 1) * max_angle
        img, mask = rotate(img, mask, random_angle)

    return img.copy(), mask.copy()


class PredictionDataset:
    def __init__(self, root: Path, size: Tuple):
        self.paths = list(sorted(root.joinpath('images').glob('*.jpg')))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image, size = load_image(path, self.size, with_size=True)
        return utils.img_transform(image), (path.stem, list(size))


def predict(model, root: Path, size: Tuple, out_path: Path, batch_size: int, workers: int):
    loader = DataLoader(
        dataset=PredictionDataset(root, size),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    threshold = 0.5
    out_path.mkdir(exist_ok=True)
    for inputs, (stems, sizes) in tqdm.tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        outputs = (model(inputs).data.cpu().numpy() > threshold).astype(np.uint8)[:, 0, :, :]
        for output, stem, width, height in zip(outputs, stems, *sizes):
            save_mask(output,
                      size=(width, height),
                      path=out_path / '{}.png'.format(stem))


def save_mask(data: np.ndarray, size: Size, path: Path):
    mask = cv2.resize(data, size, cv2.INTER_AREA) * 255
    cv2.imwrite(str(path), mask)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'],
        default='train')
    arg('--limit', type=int, help='use only N images for valid/train')
    arg('--dice-weight', type=float, default=0.0)
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--size', type=str, default='768x512',
        help='Input size, for example 768x512. Must be multiples of 32')
    utils.add_args(parser)
    args = parser.parse_args()

    root = Path(args.root)
    model = UNet11()
    if args.device_ids:
        device_ids = list(map(int, args.device_ids.split(',')))
    else:
        device_ids = None

    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    loss = Loss(dice_weight=args.dice_weight)

    w, h = map(int, args.size.split('x'))
    if not (w % 32 == 0 and h % 32 == 0):
        parser.error('Wrong --size: both dimentions should be multiples of 32')
    size = (w, h)

    if args.limit:
        limit = args.limit
        valid_limit = limit // 5
    else:
        limit = valid_limit = None

    def make_loader(ds_root: Path, limit_: int, augmentation=False):
        return DataLoader(
            dataset=StreetDataset(ds_root, size, limit=limit_, augmentation=augmentation),
            shuffle=True,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True

        )

    valid_root = utils.DATA_ROOT / 'validation'

    if args.mode == 'train':
        train_loader = make_loader(utils.DATA_ROOT / 'training', limit, augmentation=True)
        valid_loader = make_loader(valid_root, valid_limit)
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))

        utils.train(
            init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
            args=args,
            model=model,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=validation,
            patience=2,
        )

    # elif args.mode == 'valid':
    #     valid_loader = make_loader(valid_root, valid_limit)
    #     state = torch.load(str(Path(args.root) / 'model.pt'))
    #     model.load_state_dict(state['model'])
    #     validation(model, loss, tqdm.tqdm(valid_loader, desc='Validation'))
    #
    # elif args.mode == 'predict_valid':
    #     utils.load_best_model(model, root)
    #     predict(model, valid_root, out_path=root / 'validation',
    #             size=size, batch_size=args.batch_size)
    #
    elif args.mode == 'predict_test':
        utils.load_best_model(model, root)
        test_root = utils.DATA_ROOT / 'testing'
        predict(model, test_root, out_path=root / 'testing',
                size=size, batch_size=args.batch_size, workers=args.workers)


if __name__ == '__main__':
    main()
