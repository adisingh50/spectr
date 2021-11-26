"""Functions to provide I/O APIs for all the modules."""

# 1. native python libraries
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# 2. pip-installed python libraries
import gtsam
import h5py
import numpy as np

# 3. Local repository imports
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.sfm_track import SfmTrack2d


def load_image(img_path: str) -> Image:
    """Load the image from disk.

    Notes: EXIF is read as a map from (tag_id, value) where tag_id is an integer.
    In order to extract human-readable names, we use the lookup table TAGS or GPSTAGS.
    Images will be converted to RGB if in a different format.

    Args:
        img_path (str): the path of image to load.

    Returns:
        loaded image in RGB format.
    """
    original_image = PILImage.open(img_path)

    exif_data = original_image._getexif()
    if exif_data is not None:
        parsed_data = {}
        for tag_id, value in exif_data.items():
            # extract the human readable tag name
            if tag_id in TAGS:
                tag_name = TAGS.get(tag_id)
            elif tag_id in GPSTAGS:
                tag_name = GPSTAGS.get(tag_id)
            else:
                tag_name = tag_id
            parsed_data[tag_name] = value

        exif_data = parsed_data

    img_fname = Path(img_path).name
    original_image = original_image.convert("RGB") if original_image.mode != "RGB" else original_image
    return Image(value_array=np.asarray(original_image), exif_data=exif_data, file_name=img_fname)
