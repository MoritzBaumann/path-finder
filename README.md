# Path-Finder — Coastal Path Detection

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">System requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to create a mapping tool for coastal hiking paths. A pretrained image learning model (UNet) is used to identify coastline paths from geospatial imagery.

The underlying idea is to use this tool to keep mapping apps up to date and thereby increase hikers' safety along coastal pathways. It originates from Yohan Cobac, who initiated this project during the [Ocean Hackathon 2025](https://www.campusmer.fr/home-4185-0-0-0.html).

The centerpiece of the project is a pipeline consisting of 5 different steps:
- `01_preprocess`: During preprocessing images are tiled and downscaled and hiking paths (labels) are rasterized.
- `02_train`: Model training trains a pretrained UNet model on the tile/mask pairs.
- `03_visualize`: Visualization creates graphs of training metrics and preprocessed images.
- `04_evaluate`: Tests the model performance on unseen test data.
- `05_predict`: Makes hiking path predictions on new, hitherto unseen images.


<!-- GETTING STARTED -->
## Getting Started

### System requirements

   - macOS or Linux (development tested on macOS)
   - Python >=3.13
   - GPU recommended for training, but not needed

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MoritzBaumann/path-finder.git
   ```
2. Setup (and activate) your environment (example with `uv`)
  ```sh
  uv venv
  ```
3. Install dependencies (example with `uv`)
   - `uv sync`
   - -> or install core packages manually using `pip`:
   - `pip install torch rasterio ...`

<!-- USAGE EXAMPLES -->
## Usage

How this project can be used, to be filled in ...

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License.


<!-- Authors -->
## Authors

Moritz Baumann - [@LinkedIn](https://www.linkedin.com/in/moritz-baumann/)

Project Link: [https://github.com/MoritzBaumann/path-finder](https://github.com/MoritzBaumann/path-finder)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

This project was started during the [Ocean Hackathon 2025](https://www.campusmer.fr/home-4185-0-0-0.html), and the initial idea came from Yohan Cobac.

* License: [MIT License](https://choosealicense.com/licenses/mit/)