"""Online augmentation of score sheet crops for training.

Works on grey-scale uint8 crops that were pre-scaled to SOURCE_HEIGHT (larger than the network input) so that
geometric transforms are applied before the final downscale and do not smear the strokes. The error analysis of the
first sheet-level model (css-20) drives the choice of transforms:
- crop framing (margin, scale, horizontal placement) was the strongest predictor of errors,
- most errors are single-glyph shape confusions (e/c, a/g, h/b, B/R, 3/5), which stroke thickness and small
  distortions address,
- photometric variety (brightness, contrast, blur, noise) replaces the offline copies that were memorised.
"""
import math

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import maximum_filter, minimum_filter

SOURCE_HEIGHT = 48
SOURCE_WIDTH = 240


def border_colour(array):
    """Median grey level of the right border column (paper colour used for padding and affine fill)."""
    return int(np.median(array[:, -1]))


def geometric(image, rng, params):
    """Random scale (margin), rotation and shear, keeping the image height and filling with the paper colour.

    Scaling below 1 shrinks the text inside the canvas (more margin, as on sheets with tall cells); scaling above 1
    enlarges it and crops the borders (tighter Textract boxes)."""
    width, height = image.size
    scale = rng.uniform(*params['scale'])
    angle = math.radians(rng.uniform(-params['rotate'], params['rotate']))
    shear = rng.uniform(-params['shear'], params['shear'])
    new_width = max(8, round(width * scale))

    # Output -> input mapping around the image centre: inverse of (scale, then rotate, then shear).
    cos, sin = math.cos(angle), math.sin(angle)
    cx_out, cy_out = new_width / 2, height / 2
    cx_in, cy_in = width / 2, height / 2
    # Inverse affine: x_in = a*x_out + b*y_out + c, y_in = d*x_out + e*y_out + f
    a, b = cos / scale, (sin - shear * cos) / scale
    d, e = -sin / scale, (cos + shear * sin) / scale
    c = cx_in - a * cx_out - b * cy_out
    f = cy_in - d * cx_out - e * cy_out
    return image.transform((new_width, height), Image.AFFINE, (a, b, c, d, e, f),
                           resample=Image.BILINEAR, fillcolor=border_colour(np.array(image)))


def stroke_thickness(array, rng, params):
    """Thicken or thin the (dark) strokes with a min/max filter; models pen width and scan contrast."""
    roll = rng.uniform()
    if roll < params['thicken_probability']:
        return minimum_filter(array, size=rng.choice([(2, 2), (3, 1), (1, 3), (3, 3)]).tolist())
    if roll < params['thicken_probability'] + params['thin_probability']:
        return maximum_filter(array, size=rng.choice([(2, 2), (1, 3), (3, 1)]).tolist())
    return array


def photometric(array, rng, params):
    """Brightness shift, contrast change, Gaussian blur and noise."""
    array = array.astype(np.float32)
    mean = array.mean()
    array = (array - mean) * rng.uniform(*params['contrast']) + mean + rng.uniform(-1, 1) * params['brightness'] * 255
    if rng.uniform() < params['blur_probability']:
        radius = rng.uniform(*params['blur_radius'])
        array = np.array(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    if rng.uniform() < params['noise_probability']:
        array = array + rng.normal(0, rng.uniform(0, params['noise_sigma']) * 255, array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


def to_canvas(array, img_width, img_height, rng, params):
    """Downscale to the network height keeping the aspect ratio and place the text on the canvas, left-aligned
    (as the inference engine does) or at a random horizontal offset."""
    height, width = array.shape
    scaled_width = max(1, min(img_width, round(width * img_height / height)))
    array = np.array(Image.fromarray(array).resize((scaled_width, img_height), resample=Image.BILINEAR), dtype=np.uint8)
    fill = border_colour(array)
    offset = int(rng.integers(0, img_width - scaled_width + 1)) if rng.uniform() < params['offset_probability'] else 0
    return np.pad(array, ((0, 0), (offset, img_width - scaled_width - offset)), mode='constant', constant_values=fill)


def augment(source, img_width, img_height, rng, params):
    """Augment one source crop (uint8, SOURCE_HEIGHT x content width) into a network input of img_height x img_width."""
    image = geometric(Image.fromarray(source), rng, params)
    array = stroke_thickness(np.array(image, dtype=np.uint8), rng, params)
    array = photometric(array, rng, params)
    return to_canvas(array, img_width, img_height, rng, params)


def resize_to_source(image):
    """Scale a grey-scale PIL image to SOURCE_HEIGHT keeping the aspect ratio (capped at SOURCE_WIDTH).
    Returns the uint8 array padded to SOURCE_WIDTH and the content width."""
    scaled_width = max(1, min(SOURCE_WIDTH, round(image.width * SOURCE_HEIGHT / image.height)))
    array = np.array(image.resize((scaled_width, SOURCE_HEIGHT), resample=Image.BILINEAR), dtype=np.uint8)
    padded = np.pad(array, ((0, 0), (0, SOURCE_WIDTH - scaled_width)), mode='constant', constant_values=border_colour(array))
    return padded, scaled_width
