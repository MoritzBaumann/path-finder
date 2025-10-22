import yaml
import glob
from pathlib import Path
from src.data_utils import rasterize_paths, generate_tiles_from_tiff

if __name__ == "__main__":
    config_path = Path("config/data_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config['data']['raw_geotiffs_dir'])
    labels_dir = Path(config['data']['raw_labels_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    patch_size = Path(config['data']['patch_size'])
    tile_dir = Path(config['data']['tiles_dir'])

    tile_size = config['data']['tile_size']
    target_size = config['data']['target_size']
    path_pixel_dilation = config['data']['path_pixel_dilation']

    tiff_files = sorted(glob.glob(str(raw_dir / "*.tif")))
    label_files = sorted(glob.glob(str(labels_dir / "*.geojson")))
