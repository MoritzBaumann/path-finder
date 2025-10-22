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
   - Python 3.13+
   - GPU recommended for training

2. Install dependencies (example with uv)
   uv init
   uv venv
   source .venv/bin/activate
   -> or install core packages manually using pip:
   pip install torch rasterio ...

3. Prepare data
   - Place source imagery in `data/raw/geotiffs/`
   - Place label masks in `data/labels/` (geopackage file(s))
   - Use `scripts.preprocess.py` to split large GeoTIFFs into tiles for training (see notebooks for examples).

4. Preprocess & explore
   - Compute per-channel mean/std for normalization using `src.data_utils.compute_dataset_mean_std`.
   - Create tiles in `data/interim/tiles_1024x1024` (or similar).

5. Train (placeholder)
   - Implement/train by editing `src/train.py`. Example training pipeline:
     - Build Dataset using `src.data_utils.RasterTileDataset`
     - Create model in `src/model_utils.py`
     - Train, log checkpoints to `outputs/trained_models/`

6. Inference & visualization (placeholder)
   - Implement inference in `src/infer.py` to load a model and run on tiles or full rasters.
   - Visualize overlays with `src/visualize.py`.

Examples & notebooks
- notebooks/01_pytorch_training_multiple_tiffs.ipynb — example preprocessing & training workflow
- notebooks/03_pytorch_inference.ipynb — example inference steps and visualization

Evaluation
- Recommended metrics: Intersection-over-Union (IoU), F1 (per-class), pixel accuracy.
- Split data spatially to avoid leakage (do not train & test on adjacent tiles from same scene).

Contributing
- Use branches and open PRs. Add unit tests for new preprocessing/model code.
- Keep data processing deterministic and document coordinate / CRS handling when writing GeoTIFF outputs.

Notes & caveats
- Geo-referencing: when tiling/writing rasters, preserve and update GeoTransform/CRS (handled by rasterio in helper utilities).
- Nodata handling: ignore tiles dominated by nodata values during training (see tiling utility).
- Vectorization / post-processing: predicted masks can be converted to polylines for mapping apps (not implemented here).

License
- Add a LICENSE file appropriate for the intended use (e.g., MIT).

Contact
- For code structure questions or to request a specific implementation (trainer, model, inference runner), open an issue or request a specific task in this repository.