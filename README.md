# Path-Finder — Coastal Path Detection

A PyTorch-based pipeline for detecting coastline paths from geospatial (GeoTIFF) imagery. The goal is to produce pixel-wise segmentation masks that help keep mapping applications up to date and improve hiker safety.

## Features

- Preprocessing of large GeoTIFF rasters into tiles
- Dataset building for segmentation tasks
- Model training and inference with PyTorch
- Visualization of predicted coastline/path masks

## Project Structure
```
path-finder/
├── config/        # Configuration files
├── notebooks/     # Exploratory and analysis notebooks
├── scripts/       # Standalone scripts (preprocessing, inference, etc.)
├── src/           # Core source code
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python >= 3.13
- macOS or Linux
- GPU recommended for training

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install torch rasterio
```

## Usage

1. **Prepare data** — Place your GeoTIFF files in the appropriate data directory and run the preprocessing script to tile them.
2. **Train** — Use the training script or notebook to train a segmentation model on the prepared tiles.
3. **Infer** — Run inference on new imagery to produce path/coastline masks.
4. **Visualize** — Use the visualization tools to inspect predicted masks against the source imagery.

> Detailed usage instructions will be added as the project matures.

## License

[MIT](LICENSE)

## Acknowledgements

This project was started during the [Ocean Hackathon 2025](https://www.campusmer.fr/home-4185-0-0-0.html), and the initial idea came from Yohan Cobac and Laura Dréan.