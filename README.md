# Infrastructure Spectral Latent Atlas (ISLA)

This repository contains the implementation and bridge vibration datasets used for the Infrastructure Spectral Latent Atlas (ISLA), a queryable latent spectral reconstruction framework for population-level bridge structural health monitoring.

## Repository contents

- `full_script.py`: Complete ISLA implementation, including spectral preprocessing, VAE latent learning, T-CFM reconstruction, query lifting, coefficient anchoring, and visualization.
- `data/`: Compressed bridge vibration datasets.
- `outputs/`: Folder where generated results are saved.

## Data preparation

The bridge datasets are provided as compressed `.rar` files:

- `bridge1.rar`
- `bridge2.rar`
- `bridge3.rar`
- `bridge4.rar`
- `bridge5.rar`

Before running the script, extract these files into the `data/` folder so that the CSV files are available as:

```text
data/bridge1.csv
data/bridge2.csv
data/bridge3.csv
data/bridge4.csv
data/bridge5.csv
```
