from pathlib import Path

import yaml

from src.data_utils import process_geotiffs_to_tiles

if __name__ == "__main__":
    # Load configuration
    with open(Path("config/data_config.yaml")) as file:
        data_config = yaml.safe_load(file)

    geotiff_dir = Path(data_config["directories"]["raw_geotiffs_dir"])
    label_dir = Path(data_config["directories"]["raw_labels_dir"])
    interim_dir = Path(data_config["directories"]["interim_dir"])

    tile_size = data_config["tiles"]["tile_size"]
    target_size = data_config["tiles"]["target_size"]
    path_pixel_dilation = data_config["tiles"]["path_pixel_dilation"]

    process_geotiffs_to_tiles(
        geotiff_dir, label_dir, interim_dir, tile_size, target_size, path_pixel_dilation
    )
