# NDVI Tool

Computes [NDVI](https://www.earthdata.nasa.gov/topics/land-surface/normalized-difference-vegetation-index-ndvi)
from multi-band GeoTIFF satellite imagery using a
[slippy-map tile grid](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
in [WebMercator (EPSG:3857)](https://epsg.io/3857).

## What it does

1. Reads a GeoTIFF (windowed — never loads the full file into memory)
2. Reprojects it onto 512px [tiled web map](https://en.wikipedia.org/wiki/Tiled_web_map)
   tiles at a given zoom level (default 14, ≈3 m/px at 50°N — see
   [OSM zoom levels](https://wiki.openstreetmap.org/wiki/Zoom_levels))
3. Computes NDVI = (NIR − Red) / (NIR + Red) per tile, in parallel
4. Saves each tile as an individual GeoTIFF (byproduct)
5. Joins NDVI tiles into a single output GeoTIFF in WebMercator

Tiles align across input files: tile (x, y, z) from file A covers the same
geographic area as tile (x, y, z) from file B, because they share the same
global slippy-map grid.

## Input format

The tool expects [PlanetScope PSScene4Band](https://docs.planet.com/data/imagery/planetscope)
GeoTIFFs with 4 bands (1-based):

| Band | Name |
|------|------|
| 1    | Blue |
| 2    | Green |
| 3    | Red |
| 4    | Near-infrared |

NDVI uses bands 3 (Red) and 4 (NIR).

## Usage

Dependencies are [declared inline](https://packaging.python.org/en/latest/specifications/inline-script-metadata/)
in the script header ([PEP 723](https://peps.python.org/pep-0723/)).
[uv](https://docs.astral.sh/uv/) reads them automatically — no virtual
environment setup or `pip install` needed.

```bash
uv run ndvi_tool.py [-o OUTPUT_DIR] [-z ZOOM] [-w WORKERS] file1.tif [file2.tif ...]
```

Options:

| Flag | Description |
|------|-------------|
| `-o, --output` | Output directory (default: `output`) |
| `-z, --zoom` | Slippy-map zoom level (default: 14, lower for fewer/larger tiles) |
| `-w, --workers` | Parallel threads for tile processing (default: CPU count) |

Examples:

```bash
# Single file at zoom 14
uv run ndvi_tool.py -z 14 scene.tif

# Multiple files, custom output dir, 4 threads
uv run ndvi_tool.py -o results -w 4 scene_a.tif scene_b.tif
```

## Tests

```bash
uv run test_ndvi_tool.py
```

This runs the full suite (unit tests for `compute_ndvi` and
`tile_transform`, plus end-to-end CLI tests with synthetic GeoTIFFs)
in an isolated environment managed by `uv`.

## Dependencies

- [rasterio](https://rasterio.readthedocs.io/) — windowed I/O, reprojection,
  and mosaicing via GDAL (releases the GIL, so threads give real parallelism)
- [mercantile](https://pypi.org/project/mercantile/) — slippy-map tile
  coordinate math (tile enumeration, tile bounds)
- numpy — NDVI arithmetic (SIMD-accelerated element-wise operations)

## Further reading

- [NASA Earthdata — NDVI overview](https://www.earthdata.nasa.gov/topics/land-surface/normalized-difference-vegetation-index-ndvi)
- [NASA Earth Observatory — Measuring Vegetation (NDVI & EVI)](https://earthobservatory.nasa.gov/features/MeasuringVegetation/measuring_vegetation_2.php)
- [SpaceKnow docs](https://docs.spaceknow.com)
