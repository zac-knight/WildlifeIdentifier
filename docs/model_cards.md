# Model cards

Metrics for every model are in the [evaluation report](evaluation.md), generated from scripts. All
models are for non-commercial use, because their training data includes non-commercial licences.

## Image classifier

- **Intended use:** Suggesting which of 200 common Australian birds appears in a photo, as a starting
  point for people to check.
- **Not for:** Rare species records, regulatory or conservation decisions, or birds outside Australia.
- **Architecture:** EfficientNetV2-B2 from ImageNet weights, 224 px input.
- **Training data:** iNaturalist research-grade photos under CC0, CC BY and CC BY-NC. See the
  [dataset card](dataset_card.md).
- **Calibration:** Temperature scaling, stored in the model's metadata.
- **Limitations:** Closed set, so any unlisted bird must be caught by the unfamiliar-input check,
  which misses many similar birds. Trained on people's photos, not camera traps. Performance falls
  in regions and conditions unlike the training photos.
- **Ethical considerations:** Photos stay on the device. Mistakes could mislead a record if treated as
  fact, so uncertain results are never presented as identifications.

## Location and date prior

- **Intended use:** Re-ranking image and sound predictions by what is expected at a place and season.
- **Architecture:** Residual MLP on sine and cosine of latitude, longitude and day, with the SINR
  "assume negative" loss.
- **Training data:** GBIF occurrences within Australia, with held-out records removed.
- **Limitations:** Reflects where people record birds as well as where birds occur. Scores are relative
  and are not true probabilities of presence.
- **Ethical considerations:** Threatened species are trained on coordinates snapped to 0.5 degrees, so
  the public model cannot be used to rebuild a precise map of them.

## Sound classifier

- **Intended use:** Suggesting which bird is the main sound in a recording.
- **Architecture:** EfficientNetV2-B0 on log mel spectrograms computed inside the ONNX graph.
- **Training data:** iNaturalist recordings under CC0, CC BY, CC BY-NC, CC BY-SA and CC BY-NC-SA, so
  the model is also treated as share-alike.
- **Limitations:** Far fewer recordings than photos, and some species have none. Labels apply to whole
  recordings, and other species calling in the background are not labelled.

## Few-shot prototypes

- **Intended use:** Extra evidence when the classifier is unsure, for species with few examples.
- **Architecture:** Frozen DINOv2 ViT-S/14 embeddings (Apache 2.0) and mean per-species prototypes.
- **Limitations:** Whether this is enabled in the app depends on its measured result. See the report.

## Habitat models

- **Intended use:** Illustrating how climatically suitable areas may change. They are projections, not
  predictions of where a species will live.
- **Architecture:** One LightGBM model per species on 14 bioclimatic variables.
- **Training data:** GBIF occurrences in the Australasian window, with other birds' records as
  background, and climate from NASA NEX-GDDP-CMIP6 (CC0).
- **Limitations:** Climate alone at about 25 km. No land use, dispersal, fire or species interactions.
  Output is relative suitability. Future climates outside the training range are flagged.
- **Ethical considerations:** Threatened species' maps are blurred to about 100 km.

## Animal detector (not yet exported)

- **Intended use:** Finding animals, people and vehicles in camera trap images.
- **Model:** MegaDetector V6 `MDV6-mit-yolov9-c` by Microsoft. The model zoo lists it as MIT and the
  Zenodo record as CC BY 4.0.
- **Status:** The export script exists but has not been run, so camera trap mode cannot run yet.
- **Ethical considerations:** People in images are never classified or displayed by the app.
