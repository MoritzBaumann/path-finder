from pathlib import Path

import yaml

from src.visualize import visualize_tiles

if __name__ == "__main__":
    # Load configuration
    with open(Path("config/data_config.yaml")) as file:
        data_config = yaml.safe_load(file)

    processed_tile_dir = Path(data_config["directories"]["tile_dir"])

    visualize_tiles(processed_tile_dir)
