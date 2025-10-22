# Path-Finder — Coastal Path Detection

Purpose
- Create a model to identify coastline paths from geospatial imagery to help keep mapping apps up to date and increase hikers' safety.

Project overview
- PyTorch-based pipeline for preprocessing geotiffs, training segmentation models, running inference, and visualizing predicted coastline/path masks.
- Designed for tiled GeoTIFF input and mask outputs.

Key goals
- Detect coastline/path features in aerial/satellite imagery.
- Produce pixel-wise masks usable by mapping pipelines or downstream vectorization.
- Provide tools to preprocess large rasters into tiles, build datasets, train/infer models, and visualize results.

Repository layout
... to be added ...


Quick start (development)
1. System requirements
   - macOS or Linux (development tested on macOS)
   - Python >=3.13
   - GPU recommended for training

2. Install dependencies (example with uv)
   uv sync
   -> or install core packages manually using pip:
   pip install torch rasterio ...

3. Prepare data
   - ...

...