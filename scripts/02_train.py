from pathlib import Path

import yaml

from src.train import create_model_name_and_path, train_model

if __name__ == "__main__":
    # Load configurations
    with open(Path("config/data_config.yaml")) as file:
        data_config = yaml.safe_load(file)

    with open(Path("config/model_config.yaml")) as file:
        model_config = yaml.safe_load(file)

    with open(Path("config/train_config.yaml")) as file:
        train_config = yaml.safe_load(file)

    tile_dir = data_config["directories"]["tile_dir"]

    dataset_name = train_config["data_loading"]["dataset_name"]
    batch_size = train_config["data_loading"]["batch_size"]
    num_epochs = train_config["training"]["num_epochs"]
    learning_rate = train_config["training"]["learning_rate"]
    output_dir = Path(train_config["training"]["output_dir"])

    model_type = model_config["model_type"]
    encoder_name = model_config["encoder_name"]
    encoder_weights = model_config["encoder_weights"]
    in_channels = model_config["in_channels"]

    _, model_save_path = create_model_name_and_path(
        model_type,
        encoder_name,
        dataset_name,
        num_epochs,
        batch_size,
        learning_rate,
        output_dir,
        encoder_weights,
    )
    train_model(
        tile_dir,
        batch_size,
        num_epochs,
        learning_rate,
        encoder_name,
        encoder_weights,
        model_save_path,
    )
