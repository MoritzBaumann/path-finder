import yaml
from pathlib import Path
from src.visualize import visualize_tiles

if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/data_config.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    processed_tile_dir = Path(config["directories"]["tile_dir"])

    visualize_tiles(processed_tile_dir)
