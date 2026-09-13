# Changelog

## Unreleased (JOSS submission prep)

### Added
- **Spatially blocked cross-validation.** Training points are sampled every
  0.2 km along a river while each image covers ~3x3 km, so points a few
  hundred metres apart yield near-identical images. Under plain random
  k-fold these near-duplicates split across training and validation folds,
  and the model scores highly on images it has effectively already seen —
  spatial autocorrelation leakage, the standard failure mode in
  remote-sensing ML. `build_spatial_groups()` clusters points within 2 km
  (haversine, greedy single-linkage) from the lat/lon in each filename, and
  CV now uses `StratifiedGroupKFold` over those groups.
- `--no-annotation-features` flag and `exclude_annotation_features` parameter:
  trains an imagery-only model with all annotation-derived features removed.
  The labeling GUI auto-sets `label=1` when a `sand_mining` region is drawn,
  so features computed from those polygons (`sand_mining_*`, `num_*_areas`,
  `total_*_area`, `*_area_ratio` — 66 of 107 features in a typical run)
  partly encode the label itself. A model using them can score 100% without
  learning anything about imagery. Scores from the leakage-free model are the
  ones valid to report for detection performance on unlabeled imagery.
- Automatic target-leakage diagnostic (`report_leakage_risk`): before
  training, prints any annotation-derived feature correlating |r| >= 0.7
  with the label.
- Cross-validated metrics (F1, ROC-AUC, PR-AUC, balanced accuracy) in the
  single-model path, which previously reported only a small hold-out split.
  Fold count adapts to minority-class size. A warning is printed when the
  hold-out set is under 20 samples.
- `--mode retrain`: train on labels that already exist on disk, without
  re-downloading imagery or re-opening the labeling GUI. Previously the only
  route to a trained model was `--mode train`, which always re-downloaded
  images and forced a pass through the GUI, so there was no supported way to
  rebuild a model after editing labels.
- Single-image analysis mode (`--mode analyze-image`): scores an arbitrary
  image for sand-mining likelihood, produces a SLIC-superpixel heatmap
  overlay, and a structured JSON list of scored regions.
- Optional frozen pretrained diffusion-transformer (DiT) deep-feature
  extractor (`--use-dit` / `config.USE_DIT_FEATURES`), concatenated onto the
  existing spectral/texture feature vector (`src/dit_features.py`).
- Checkpointing/resumability for all long-running loops: training-image
  download, feature extraction (with and without historical trends),
  river-point probability mapping, and scraper geocoding. Each now saves
  progress to `outputs/checkpoints/` and resumes automatically on restart.
- Test suite (`tests/`) covering fusion math, feature-extraction helpers,
  and checkpoint utilities (no Earth Engine dependency required to run).
- Packaging: `pyproject.toml`, `requirements.txt`, `requirements-dit.txt`,
  `requirements-dev.txt`, MIT `LICENSE`.

### Fixed
- `--mode retrain` never initialized Earth Engine, so every
  `get_historical_images()` call failed with "Earth Engine client library not
  initialized" and all historical trend features (`NDVI_trend`, `NDWI_trend`,
  `MNDWI_trend`, `BSI_trend`, `hist_periods`) were silently written as zeros —
  a run appeared to succeed while quietly training on image features alone.
  EE is now initialized when historical features are enabled.
- `src/scraper.py`: `geocode_all()` was called from `run_scraper()` but its
  function definition had been lost — an orphaned, unindented code block
  with the same logic was left dangling in the module, causing a `NameError`
  on any non-cached scraper run. Restored as a proper function.
- `src/mapper.py`: `SandMiningProbabilityMapper.__init__` accepted
  `model_path`/`scaler_path` arguments but never used them, always reloading
  from `config.DEFAULT_MODEL_FILE`/`DEFAULT_SCALER_FILE` regardless of what
  was passed. `run_mapping_workflow()` was also passing already-loaded
  model/scaler *objects* positionally where the constructor expected file
  *paths*. Both fixed.
- `src/model.py` (data leakage): `StandardScaler` was fit on the **full**
  dataset before the train/test split in both `train_model` and
  `train_multiple_models`, letting test-set statistics influence training.
  The split now happens first and the scaler is fit on the training portion
  only. Cross-validation likewise ran on pre-scaled data; scaling and SMOTE
  are now applied inside each fold via a pipeline.
- `scripts/run_pipeline.py`: the tkinter labeling GUI was imported at module
  level, so on any headless system (server, container, CI) the entire CLI
  failed at startup with `ModuleNotFoundError: No module named 'tkinter'` —
  even for modes that never open a GUI (`map`, `analyze-image`, `full`,
  `all-river`). The GUI is now imported lazily, inside the labeling function,
  with a clear error message if it is unavailable.
- `src/ee_utils.py`: removed a hardcoded personal Google Cloud project ID;
  now reads `EE_PROJECT_ID` from the environment (or `config.EE_PROJECT_ID`),
  failing with a clear message if unset, so the tool is installable and
  runnable by anyone with their own Earth Engine project.

## Prior versions

Prior history predates this changelog; see git log.
