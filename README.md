# wire_catenary

Detect overhead wires in drone LiDAR point clouds and fit 3D catenary curves to each one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the package + deps
# or just: pip install -r requirements.txt
```

Place the `.parquet` data files in the project root (they're gitignored).

## Usage

```bash
# CLI
python -m wire_catenary lidar_cable_points_easy.parquet
python -m wire_catenary lidar_cable_points_hard.parquet --plot
python -m wire_catenary lidar_cable_points_medium.parquet --out results.csv

# batch all datasets
python scripts/run_case.py --plot

# notebook (full walkthrough with plots)
jupyter notebook lidar_catenary_assignment.ipynb

# tests
pytest tests/
```

Python API:

```python
from wire_catenary import run_dataset

for r in run_dataset("lidar_cable_points_easy.parquet"):
    print(r)
```

## How it works

The pipeline has four stages:

1. **Load + preprocess** — read x/y/z from parquet, drop outliers (per-axis z-score > 3.5), optionally downsample.

2. **Cluster** — DBSCAN on the (x, y) plane. Epsilon is estimated automatically from the 90th-percentile of k-NN distances, so it adapts to point density. I chose DBSCAN because we don't know the wire count up front and it handles noise natively. If the first pass finds nothing, a second pass with relaxed params is tried.

3. **Fit per wire** — For each cluster:
   - SVD/PCA gives the dominant direction (along the wire).
   - A vertical plane through that direction defines local (u, v) coordinates — u along the wire, v for height/sag.
   - Fit the catenary `v = v0 + c·cosh((u − u0)/c) − c` using `scipy.optimize.least_squares` with soft_l1 loss for robustness.

4. **Reconstruct + report** — map the fitted 2D curve back to 3D via `P(u) = centroid + u·d + v(u)·up`. Output fitted parameters, RMSE, and optional CSV/plots.

## Project layout

```
src/wire_catenary/
    io.py              parquet → (N,3) array
    preprocessing.py   outlier removal, downsampling
    clustering.py      DBSCAN with auto eps
    plane_fit.py       PCA frame + projection
    catenary.py        model, fitting, 3D reconstruction
    pipeline.py        ties it all together
    visualize.py       matplotlib plots
    cli.py             argparse entry point
tests/
    test_catenary.py   catenary evaluate/fit tests
    test_plane_fit.py  PCA direction + projection tests
scripts/
    run_case.py        batch runner for all datasets
```

## Results summary

- **easy** — 10 wires, all RMSE < 0.04 m. Clean separation, catenary fits well.
- **hard** — 1 wire, RMSE ~0.04 m. Still good despite added noise.
- **extrahard** — 1 cluster (nearby wires merge). RMSE ~0.04 m on what it does find.
- **medium** — 1 cluster, high RMSE (~1.7 m). Background clutter is lumped with wire points — per-cluster outlier rejection would help most here.

## Limitations

- Clustering on (x, y) only — near-vertical wires won't separate.
- The auto eps works well on the provided datasets but might need tuning on very different point densities.
- Each wire is assumed to sag in a single vertical plane.
- The medium dataset result is poor: the main issue is that DBSCAN lumps noise with wire points. Per-cluster outlier rejection before fitting would help a lot here.

## What I'd do next

- Reject outliers within each cluster before fitting (biggest impact on medium/hard cases)
- Try HDBSCAN — it's better at varying-density clusters
- Multiple optimizer restarts to avoid local minima
- Confidence scoring to flag wires where the catenary model doesn't fit well