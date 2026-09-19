# Syntropy Digital — Geospatial Automated Raster Feature Pipeline

A high-throughput, memory-efficient Python CLI tool for automated geological ridge, structural fault, and linear boundary extraction from multi-spectral remote sensing GeoTIFFs.

## Key Capabilities
- **Chunked Window Processing:** Implements Rasterio block windows ($512 \times 512$) allowing multi-gigabyte rasters (10+ GB) to be processed with constant low RAM (<500 MB).
- **Multi-Scale Gradient Filtering:** Combines Gaussian smoothing with Sobel directional derivative kernels to isolate linear geological boundaries while suppressing sensor noise.
- **Topology-Preserving Vectorization:** Generates clean, closed polygon and linestring geometries exported directly to standards-compliant GeoJSON with preserved Coordinate Reference Systems (CRS).

## Installation & Dependencies
```bash
pip install rasterio geopandas shapely scipy numpy
```

## Usage
```bash
python raster_feature_extractor.py \
  --input path/to/multispectral_scene.tif \
  --out-tif output_features.tif \
  --out-geojson output_features.geojson \
  --band 1 \
  --threshold 0.65 \
  --min-area 25
```

## Outputs
1. **`output_features.tif`**: Single-band binary GeoTIFF mask with DEFLATE compression.
2. **`output_features.geojson`**: Vectorized feature polygons with attributes (`feature_id`, `pixel_area`) ready for import into QGIS, ArcGIS Pro, or Mapbox.
