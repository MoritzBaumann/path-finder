import yaml
from pathlib import Path
from src.data_utils import process_geotiffs_to_tiles

if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/data_config.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    geotiff_dir = Path(config["data"]["raw_geotiffs_dir"])
    label_dir = Path(config["data"]["raw_labels_dir"])
    interim_dir = Path(config["data"]["interim_dir"])

    tile_size = config['tiles']['tile_size']
    target_size = config['tiles']['target_size']
    path_pixel_dilation = config['tiles']['path_pixel_dilation']

    process_geotiffs_to_tiles(
        geotiff_dir, label_dir, interim_dir, tile_size, target_size, path_pixel_dilation)
