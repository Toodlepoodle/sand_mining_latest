# Sand Mining Detector

A multi-layer satellite-imagery pipeline for detecting and mapping riverbed
sand mining, combining Google Earth Engine imagery, weak supervision from
scraped news/OSM/government sources, unsupervised spectral scoring, and
supervised machine learning (Random Forest / XGBoost / LightGBM), with an
optional frozen diffusion-transformer deep-feature extractor layered on top.

## Features

- **Multi-source satellite imagery** (Sentinel-2, Landsat 5/7/8/9, Sentinel-1
  SAR, VIIRS fallback) via Google Earth Engine.
- **Interactive labeling GUI** for drawing free-hand mining-area annotations
  on downloaded training images.
- **Three-layer fusion pipeline** (`--mode full`):
  1. Unsupervised spectral scoring (BSI/CMI/NDVI/MNDWI, no labels needed)
  2. Weak supervision from an 18-source scraped ground-truth database
     (OpenStreetMap, Wikidata, USGS, news outlets, NGT orders, etc.)
  3. Supervised ML on labeled + annotated training images
  — combined via automatically-tuned fusion weights.
- **Historical trend features**: multi-year quarterly NDVI/BSI/NDWI trend
  slopes per point, for detecting gradual riverbank change.
- **Single-image analysis** (`--mode analyze-image`): given one arbitrary
  image, produces an overall 0-1 sand-mining probability score, a heatmap
  overlay highlighting suspected areas (via SLIC superpixel segmentation),
  and a structured JSON list of scored regions.
- **Optional diffusion-transformer (DiT) deep features** (`--use-dit`): a
  frozen, pretrained DiT backbone is used purely as a feature extractor —
  its pooled intermediate representations are concatenated onto the
  existing spectral/texture feature vector before classification. No
  diffusion model is trained; see [Methodology notes](#methodology-notes).
- **Resumable / checkpointed long-running steps**: training-image download,
  feature extraction (with and without historical trends), river-point
  mapping, and geocoding all save progress to disk and pick up where they
  left off if interrupted, instead of restarting from scratch.
- **Interactive Folium output maps** with risk-tier marker clusters and a
  probability heatmap layer.

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/sand-mining-detector.git
cd sand-mining-detector
pip install -r requirements.txt
# Optional, only needed for --use-dit:
pip install -r requirements-dit.txt
```

Or as an installable package:

```bash
pip install -e .
pip install -e ".[dit]"   # optional DiT extra
```

### Earth Engine setup

This tool requires a Google Cloud project with the Earth Engine API enabled.

```bash
gcloud auth application-default login
export EE_PROJECT_ID=your-gcp-project-id
```

`EE_PROJECT_ID` must be set (via environment variable) before running any
mode that touches Earth Engine (`train`, `map`, `both`, `full`).

## Usage

```bash
# Train a model for one river from a shapefile
python scripts/run_pipeline.py --mode train --shapefile rivers/ganga.shp

# Rebuild the model from labels you already have (no re-download, no GUI)
python scripts/run_pipeline.py --mode retrain --shapefile rivers/ganga.shp

# Generate a probability map with an existing model
python scripts/run_pipeline.py --mode map --shapefile rivers/ganga.shp

# Full three-layer fusion pipeline
python scripts/run_pipeline.py --mode full --shapefile rivers/ganga.shp

# Score a single arbitrary image and mark suspected areas
python scripts/run_pipeline.py --mode analyze-image --image path/to/image.png

# Any of the above with diffusion-transformer deep features enabled
python scripts/run_pipeline.py --mode train --shapefile rivers/ganga.shp --use-dit
```

Run `python scripts/run_pipeline.py --help` for the full argument list.

### Resuming an interrupted run

Every long-running step (image download, feature extraction, river-point
mapping, geocoding) checkpoints its progress under
`outputs/checkpoints/`. If a run is interrupted (Ctrl-C, network failure,
Earth Engine rate limit, crash), simply re-run the same command — it will
skip everything already completed instead of starting over.

## Methodology notes

- **Single-image region scoring is an approximation.** The classifiers are
  trained on features computed over whole training images (optionally with
  drawn area-annotation features). For `--mode analyze-image`, each SLIC
  superpixel is scored by computing the same *global-style* spectral/texture
  feature set on that crop and zero-filling any annotation-derived columns
  the model expects but that arbitrary unseen images don't have. This
  surfaces relative regions of concern within an image rather than a
  pixel-perfect trained segmentation mask.
- **DiT features are frozen, not trained.** A full Diffusion Transformer is
  a generative architecture; training or fine-tuning one end-to-end would
  need far more labeled sand-mining imagery than this project has (training
  is gated on as few as ~10 labeled images). Instead, a large pretrained DiT
  backbone is used purely as a fixed feature extractor — an image is
  VAE-encoded, noised to a fixed timestep, passed once through the frozen
  DiT, and its pooled intermediate activations are appended to the existing
  hand-engineered feature vector. See `src/dit_features.py` for details and
  references.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover the Earth-Engine-independent logic: fusion math, feature-helper
functions, and the checkpoint/resume utilities.

## Citation

If you use this tool in your research, please cite it — see `CITATION.cff`
(or the JOSS paper once published).

## License

MIT — see [LICENSE](LICENSE).
