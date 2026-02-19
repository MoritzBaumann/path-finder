from pathlib import Path

import yaml

from src.utils import load_latest_history
from src.visualize import plot_training_curves, visualize_tiles

if __name__ == "__main__":
    # Load configurations
    with open(Path("config/data_config.yaml")) as file:
        data_config = yaml.safe_load(file)
    with open(Path("config/train_config.yaml")) as file:
        train_config = yaml.safe_load(file)

    processed_tile_dir = Path(data_config["directories"]["tile_dir"])
    model_output_dir = Path(train_config["training"]["model_output_dir"])
    plots_output_dir = Path(train_config["training"]["plots_output_dir"])

    visualize_tiles(processed_tile_dir)

    history, history_filename = load_latest_history(model_output_dir)

    model_run_name = history_filename.rsplit(".")[0].rstrip("_history")

    plot_training_curves(history, plots_output_dir, model_run_name)
