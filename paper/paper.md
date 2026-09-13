---
title: 'Sand Mining Detector: A Multi-Layer Satellite Imagery Pipeline for Detecting and Mapping Riverbed Sand Mining'
tags:
  - Python
  - remote sensing
  - Google Earth Engine
  - environmental monitoring
  - machine learning
  - weak supervision
  - rivers
authors:
  - name: Your Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Your Institution
    index: 1
date: 07 September 2026
bibliography: paper.bib
---

# Summary

`sand-mining-detector` is a Python pipeline that detects and maps riverbed
sand mining activity from freely available satellite imagery. Unregulated
in-stream sand extraction degrades river morphology, undermines
infrastructure, and destroys aquatic habitat, but it is geographically
diffuse and rarely monitored systematically, particularly in low- and
middle-income countries where enforcement capacity is limited. The tool
combines three complementary detection layers — unsupervised spectral
scoring, weak supervision from openly scraped mining-location evidence, and
supervised machine learning on user-labeled imagery — into a single fused
probability per location along a river, and produces an interactive map of
results. It also provides a single-image inference mode that scores an
arbitrary image and highlights suspected mining areas, and an optional deep
feature extractor based on a frozen, pretrained diffusion transformer (DiT)
that can be layered on top of the existing hand-engineered features to
improve classification accuracy without requiring additional labeled data
or end-to-end deep-model training.

# Statement of need

Detecting illegal or unregulated sand mining from satellite imagery has been
studied with unsupervised spectral methods [@mukherjee2023], with weakly
supervised approaches that exploit auxiliary evidence such as news reports
and crowd-sourced locations, and with supervised classifiers trained on
hand-labeled imagery [@gallwey2020; @li2024]. Prior work in this space,
however, tends to implement a single one of these strategies in isolation,
as a research script rather than an installable, documented, tested tool.
Practitioners — river-basin authorities, environmental NGOs, and journalists
covering illegal extraction — need a tool that (1) works with no manually
labeled data at all as a starting point (the unsupervised layer), (2)
improves automatically as openly available evidence accumulates (the weak
supervision layer), (3) improves further as a user labels a modest number of
example images (the supervised layer), and (4) combines all three
consistently rather than requiring a practitioner to choose one and discard
the others. `sand-mining-detector` is designed around this progression: a
user can produce a first probability map with zero labeled images, and
improve it incrementally as they invest more labeling effort, without
switching tools.

The package also addresses a practical obstacle specific to this domain:
sand-mining classification and mapping workflows can involve downloading and
analyzing thousands of satellite image patches per river, each requiring one
or more Earth Engine API calls, geocoding queries, or feature-extraction
passes that individually take seconds. A single end-to-end run can take
hours, and Earth Engine imposes rate limits that make transient failures
common. Every long-running step in the pipeline (training-image download,
per-image feature extraction with optional multi-year historical trend
computation, per-point probability mapping along a river, and geocoding of
scraped location mentions) is therefore checkpointed to disk and resumable,
so an interrupted run picks up where it left off rather than restarting.

Finally, the package extends its supervised layer with an optional deep
feature extractor built on a frozen, pretrained diffusion transformer
[@peebles2023]. Rather than fine-tuning a diffusion transformer end-to-end —
which would require substantially more labeled imagery than is typically
available for a given river — the backbone's intermediate representations
are extracted at a fixed noise timestep and pooled into a feature vector
that is concatenated with the existing spectral, texture, and GLCM features
before classification, following the general approach of using diffusion
model internals as learned visual representations [@xiang2023]. This lets
users benefit from representations learned from large-scale pretraining
without needing enough labeled sand-mining imagery to train a deep model
from scratch.

# Functionality

The pipeline exposes a single command-line entry point (`run_pipeline.py`)
with operating modes for: downloading and labeling training imagery via an
interactive GUI with free-hand region annotation; training a per-river or
pooled all-river classifier; generating an interactive probability map along
a river from a shapefile; running the full three-layer fusion pipeline with
automatically tuned layer weights; and scoring a single arbitrary image,
returning an overall probability, a superpixel-based heatmap overlay, and a
structured list of scored regions. Weights for fusing the three detection
layers are tuned automatically via constrained optimization against
available validation points (scraped locations and/or manually labeled
points), and are cached and reused across runs.

# Acknowledgements

We acknowledge the maintainers of Google Earth Engine, scikit-learn,
scikit-image, and the other open-source packages this tool depends on.

# References
