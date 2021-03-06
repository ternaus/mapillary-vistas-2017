#!/usr/bin/env python3
import argparse
import json
import gzip
from pathlib import Path
import shutil
from typing import Dict, Tuple
import warnings

import numpy as np
from PIL import Image
import skimage.io
import skimage.exposure
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import tqdm

import dataset
from unet_models import Loss, LinkNet34
import utils
import cv2


Size = Tuple[int, int]


class StreetDataset(Dataset):
    def __init__(self, root: Path, size: Size, limit=None):
        self.image_paths = sorted(root.joinpath('images').glob('*.jpg'))
        self.mask_paths = sorted(root.joinpath('instances').glob('*.png'))
        if limit:
            self.image_paths = self.image_paths[:limit]
            self.mask_paths = self.mask_paths[:limit]
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx], size=self.size)
        mask = load_mask(self.mask_paths[idx], size=self.size)
        return utils.img_transform(img), torch.from_numpy(mask)


def load_image(path: Path, size: Size, with_size: bool=False):
    # cached_path = path.parent / '{}-{}'.format(*size) / '{}.jpg'.format(path.stem)
    # if not cached_path.parent.exists():
    #     cached_path.parent.mkdir()
    # if cached_path.exists():
    #     image = utils.load_image(cached_path)
    # else:
    image = utils.load_image(path)
    image = image.resize(size, resample=Image.BILINEAR)
        # image.save(str(cached_path))
    if with_size:
        size = Image.open(str(path)).size
        return image, size
    else:
        return image


def load_mask(path: Path, size: Size):
    mask = Image.fromarray(cv2.imread(str(path), 0))

    mask = np.array(mask.resize(size, resample=Image.NEAREST), dtype=np.int64)
    return mask


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []
    confusion_matrix = np.zeros(
        (dataset.N_CLASSES, dataset.N_CLASSES), dtype=np.uint32)
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        output_classes = outputs.data.cpu().numpy().argmax(axis=1)
        target_classes = targets.data.cpu().numpy()
        confusion_matrix += calculate_confusion_matrix_from_arrays(
            output_classes, target_classes, dataset.N_CLASSES)
    valid_loss = np.mean(losses)  # type: float
    ious = {'iou_{}'.format(cls): iou
            for cls, iou in enumerate(calculate_iou(confusion_matrix))}
    average_iou = np.mean(list(ious.values()))
    print('Valid loss: {:.4f}, average IoU: {:.4f}'.format(valid_loss, average_iou))
    metrics = {'valid_loss': valid_loss, 'iou': average_iou}
    metrics.update(ious)
    return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


class PredictionDataset:
    def __init__(self, root: Path, size: Size):
        self.paths = list(sorted(root.joinpath('images').glob('*.jpg')))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image, size = load_image(path, self.size, with_size=True)
        return utils.img_transform(image), (path.stem, list(size))


def predict(model, root: Path, size: Size, out_path: Path, batch_size: int):
    loader = DataLoader(
        dataset=PredictionDataset(root, size),
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )
    model.eval()
    out_path.mkdir(exist_ok=True)
    for inputs, (stems, sizes) in tqdm.tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        outputs = np.exp(model(inputs).data.cpu().numpy())
        for output, stem, width, height in zip(outputs, stems, *sizes):
            save_mask(output.argmax(axis=0).astype(np.uint8),
                      size=(width, height),
                      path=out_path / '{}.png'.format(stem))


_palette = None


def get_palette():
    global _palette
    if _palette is None:
        mask = Image.open(
            str(next((utils.DATA_ROOT / 'training' / 'labels').glob('*.png'))))
        _palette = mask.getpalette()
    return _palette


def save_mask(data: np.ndarray, size: Size, path: Path):
    assert data.dtype == np.uint8
    h, w = data.shape
    mask_img = Image.frombuffer('P', (w, h), data, 'raw', 'P', 0, 1)
    mask_img.putpalette(get_palette())
    mask_img = mask_img.resize(size, resample=Image.NEAREST)
    mask_img.save(str(path))


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
    model = LinkNet34()
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

    def make_loader(ds_root: Path, limit_: int):
        return DataLoader(
            dataset=StreetDataset(ds_root, size, limit=limit_),
            shuffle=True,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True

        )
    valid_root = utils.DATA_ROOT / 'validation'

    if args.mode == 'train':
        train_loader = make_loader(utils.DATA_ROOT / 'training', limit)
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

    elif args.mode == 'valid':
        valid_loader = make_loader(valid_root, valid_limit)
        state = torch.load(str(Path(args.root) / 'model.pt'))
        model.load_state_dict(state['model'])
        validation(model, loss, tqdm.tqdm(valid_loader, desc='Validation'))

    elif args.mode == 'predict_valid':
        utils.load_best_model(model, root)
        predict(model, valid_root, out_path=root / 'validation',
                size=size, batch_size=args.batch_size)

    elif args.mode == 'predict_test':
        utils.load_best_model(model, root)
        test_root = utils.DATA_ROOT / 'testing'
        predict(model, test_root, out_path=root / 'testing',
                size=size, batch_size=args.batch_size)


if __name__ == '__main__':
    main()