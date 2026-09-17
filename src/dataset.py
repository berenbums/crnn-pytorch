import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from augment import SOURCE_HEIGHT, SOURCE_WIDTH, augment, resize_to_source


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


def open_grey(path, img_width, img_height):
    """Open an image as grey-scale; corrupt images become blank crops of the requested size."""
    try:
        return Image.open(path).convert('L') # grey-scale
    except IOError:
        print(f'Corrupted image {path}')
        return Image.new('L', (img_width, img_height), 255)


def load_image_array(path, img_width, img_height):
    """Decode an image to a grey-scale uint8 array of the network input size (deterministic, as in inference)."""
    return resize_keep_aspect(open_grey(path, img_width, img_height), img_width, img_height)


def load_source_array(path):
    """Decode an image to the larger source size used for online augmentation; returns (array, content width)."""
    return resize_to_source(open_grey(path, SOURCE_WIDTH, SOURCE_HEIGHT))


class CssDataset(Dataset):
    CHARS = ' 0123456789abcdefghPNBRQKSLTDCAFHZJVGWŞOnqrEix+-=()/!?#.:'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100, cache_dir=None, augmentation=None):
        """Dataset of score sheet crops.

        With cache_dir, all crops of a split are decoded once into a uint8 array stored under cache_dir and
        memory-mapped afterwards, so training epochs do not pay for JPEG decoding. Keep the cache on a local disk.
        With augmentation (a parameter dict, training only), crops are cached at the larger source size
        ('<mode>_src_<H>x<W>.npy' plus content widths) and augmented online in __getitem__."""
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.augmentation = augmentation
        self.images, self.widths = None, None
        self._rng = None
        if cache_dir and mode:
            self.images, self.widths = self._load_cache(cache_dir, mode)

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
        """Return (images, widths) memory-mapped from the cache for this split, building it with all CPU cores if
        missing. widths is None for the network-input cache and the content width per crop for the source cache."""
        if self.augmentation:
            cache_file = os.path.join(cache_dir, f'{mode}_src_{SOURCE_HEIGHT}x{SOURCE_WIDTH}.npy')
        else:
            cache_file = os.path.join(cache_dir, f'{mode}_{self.img_height}x{self.img_width}.npy')
        widths_file = cache_file.replace('.npy', '_widths.npy')

        if os.path.isfile(cache_file) and (not self.augmentation or os.path.isfile(widths_file)):
            images = np.load(cache_file, mmap_mode='r')
            if len(images) == len(self.paths):
                return images, np.load(widths_file) if self.augmentation else None
            print(f'Image cache {cache_file} holds {len(images)} images, expected {len(self.paths)}: rebuilding')

        print(f'Caching {len(self.paths)} images to {cache_file}...')
        os.makedirs(cache_dir, exist_ok=True)
        with Pool() as pool:
            if self.augmentation:
                arrays, widths = zip(*pool.map(load_source_array, self.paths, chunksize=256))
                np.save(widths_file, np.array(widths, dtype=np.int16))
            else:
                loader = partial(load_image_array, img_width=self.img_width, img_height=self.img_height)
                arrays = pool.map(loader, self.paths, chunksize=256)
        np.save(cache_file, np.stack(arrays))
        return np.load(cache_file, mmap_mode='r'), np.load(widths_file) if self.augmentation else None

    def sample_weights(self, collection_weights, rare_char_threshold, rare_char_weight):
        """Per-sample weights for a WeightedRandomSampler: the collection weight of the crop, multiplied by
        rare_char_weight when the label contains a character that occurs fewer than rare_char_threshold times."""
        char_counts = Counter(''.join(self.texts))
        rare_chars = {char for char, count in char_counts.items() if count < rare_char_threshold}
        print(f'rare characters (< {rare_char_threshold} occurrences): {"".join(sorted(rare_chars))}')
        weights = []
        for path, text in zip(self.paths, self.texts):
            collection = os.path.basename(os.path.dirname(path)).replace('-aug', '').rstrip('-0123456789')
            weight = collection_weights.get(collection, 1.0)
            if any(char in rare_chars for char in text):
                weight *= rare_char_weight
            weights.append(weight)
        return weights

    def __len__(self):
        return len(self.paths)

    def _input_array(self, index):
        """Network input (uint8, img_height x img_width) for one crop: augmented from the source cache in training,
        deterministic resize otherwise."""
        if self.augmentation:
            if self._rng is None:
                # One generator per worker process, seeded from the OS so that workers do not repeat each other.
                self._rng = np.random.default_rng()
            if self.images is not None:
                source = self.images[index][:, :self.widths[index]]
            else:
                source, width = load_source_array(self.paths[index])
                source = source[:, :width]
            return augment(np.ascontiguousarray(source), self.img_width, self.img_height, self._rng, self.augmentation)
        if self.images is not None:
            return self.images[index]
        return load_image_array(self.paths[index], self.img_width, self.img_height)

    def __getitem__(self, index):
        image = np.array(self._input_array(index), dtype=np.float32).reshape((1, self.img_height, self.img_width))
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
