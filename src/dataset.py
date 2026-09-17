import os
from functools import partial
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def resize_keep_aspect(image, img_width, img_height):
    """Scale a grey-scale image to the target height keeping its aspect ratio, then pad it on the right to the
    target width with the colour of its right border (paper). Crops wider than the target are squashed to fit.

    This keeps every character at its natural width, unlike a plain resize that squashes a 350x80 crop into 100x32."""
    scaled_width = max(1, min(img_width, round(image.width * img_height / image.height)))
    image = image.resize((scaled_width, img_height), resample=Image.BILINEAR)
    array = np.array(image, dtype=np.uint8)
    if scaled_width == img_width:
        return array
    fill = int(np.median(array[:, -1]))
    return np.pad(array, ((0, 0), (0, img_width - scaled_width)), mode='constant', constant_values=fill)


def load_image_array(path, img_width, img_height):
    """Decode an image to a grey-scale uint8 array of the configured size. Corrupt images become blank crops."""
    try:
        image = Image.open(path).convert('L') # grey-scale
    except IOError:
        print(f'Corrupted image {path}')
        return np.zeros((img_height, img_width), dtype=np.uint8)
    return resize_keep_aspect(image, img_width, img_height)


class CssDataset(Dataset):
    CHARS = ' 0123456789abcdefghPNBRQKSLTDCAFHZJVGWŞOnqrEix+-=()/!?#.:'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100, cache_dir=None):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        """With cache_dir, all crops of a split are decoded and resized once into a uint8 array stored as
        '<cache_dir>/<mode>_<height>x<width>.npy' and memory-mapped afterwards, so training epochs do not pay for
        JPEG decoding. Keep the cache on a local disk (not a network drive) for fast random access."""
        self.images = self._load_cache(cache_dir, mode) if cache_dir and mode else None

    def _load_from_raw_files(self, root_dir, mode):
        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                if not line.startswith('#'):
                    line_structure = line.strip().split(' ')
                    path = os.path.join(root_dir, line_structure[0])
                    text = ' '.join(line_structure[1:])

                    paths.append(path)
                    texts.append(text)
        return paths, texts

    def _load_cache(self, cache_dir, mode):
        """Return the memory-mapped image cache for this split, building it with all CPU cores if it is missing."""
        cache_file = os.path.join(cache_dir, f'{mode}_{self.img_height}x{self.img_width}.npy')
        if os.path.isfile(cache_file):
            images = np.load(cache_file, mmap_mode='r')
            if len(images) == len(self.paths):
                return images
            print(f'Image cache {cache_file} holds {len(images)} images, expected {len(self.paths)}: rebuilding')

        print(f'Caching {len(self.paths)} images to {cache_file}...')
        loader = partial(load_image_array, img_width=self.img_width, img_height=self.img_height)
        with Pool() as pool:
            images = np.stack(pool.map(loader, self.paths, chunksize=256))
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_file, images)
        return np.load(cache_file, mmap_mode='r')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.images is not None:
            image = self.images[index]
        else:
            image = load_image_array(self.paths[index], self.img_width, self.img_height)

        image = np.array(image, dtype=np.float32).reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.from_numpy(image)
        if self.texts:
            text = self.texts[index]
            try:
                target = [self.CHAR2LABEL[c] for c in text]
            except KeyError:
                raise KeyError(f'Caught KeyError for text {text}')
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def css_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
