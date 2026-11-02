# Decision log

Each entry records a significant choice, the alternatives considered and the reason.

## D-001: Region and taxon are Australia and birds

- **Decision:** Version 1 covers Australian birds (iNaturalist place 6744, GBIF `country=AU`, class Aves).
- **Alternatives:** Mammals would suit camera traps better but have far less audio.
- **Reason:** Owner's choice. Birds are well served by both photos and sound. The known cost is that
  camera trap images, which are mostly mammals, are a poor match for a bird model.

## D-002: Every data source must be free, need no login and grant reuse

- **Decision:** All data comes from sources that need no account and whose licences permit reuse.
- **Alternatives:** The original brief named WorldClim, Xeno-canto, the IUCN token API and GBIF
  bulk downloads.
- **Reason:** Owner's instruction. WorldClim bars redistribution without permission, and the others
  need an account or key.

| Need | Source used | Replaces |
| --- | --- | --- |
| Photos | iNaturalist API, with files from its AWS open data bucket | Same |
| Audio | iNaturalist sounds with Creative Commons licences | Xeno-canto |
| Occurrences | GBIF occurrence search API | GBIF account downloads |
| Conservation status | GBIF `species/{key}/iucnRedListCategory` | IUCN token API |
| Climate | NASA NEX-GDDP-CMIP6 on AWS Open Data (CC0) | WorldClim |

- **Scope note:** The owner gave this rule when answering the climate question. It is applied to all
  data sources because the no-login routes exist and no keys were supplied. Hosting accounts
  (Vercel, Neon, Hugging Face) and the optional LLM are not data sources and still need the owner's
  sign-in. Without an LLM key, species cards show the retrieved facts directly.
- **Unverified:** iNaturalist's recommended practices and terms pages return 403 to automated
  requests. Rate limits follow staff forum guidance of about 1 request per second and 10,000 per day.
  Individual media licences are honoured file by file.

## D-003: Train locally on Apple silicon, keeping scripts portable to Kaggle

- **Decision:** Training scripts are plain CLIs configured by environment variables and run locally
  on the M4 GPU (MPS).
- **Alternatives:** Kaggle or Colab free GPUs need the owner's sign-in.
- **Reason:** This allows the build to proceed without credentials. Runs are slower, and the scripts
  still work on a free GPU because nothing depends on the machine.

## D-004: EfficientNetV2-B2 as the browser classifier

- **Decision:** Fine-tune `tf_efficientnetv2_b2` from ImageNet weights at 224 px.
- **Alternatives measured on the M4 (training images per second, parameters):** ConvNeXt Tiny (23,
  28M), EfficientNetV2-S (36, 20M), ConvNeXt Pico (58, 8.6M), B2 (72, 9M), B0 (106, 6.1M).
- **Reason:** At 9M parameters it is about 18 MB at fp16. Its global pool feeds one linear layer, so
  its CAM is exact, and a test proves this. ConvNeXt normalises after pooling, so its map would be
  an approximation. DINOv2 with a linear probe is the comparison baseline.

## D-005: Training choices

- **Balanced sampling:** Each species is equally likely per draw, approximating a flat class prior.
  This is what makes it valid to multiply by the location prior.
- **Mild colour jitter:** Plumage colour separates species, so strong colour changes destroy signal.
- **Label smoothing 0.1:** Discourages overconfidence on hard pairs. Temperature scaling then
  corrects the remaining miscalibration.
- **Early stopping on macro accuracy:** Common species cannot mask a decline elsewhere.
- **Experiment tracking:** Each run writes `config.json` and `metrics.jsonl`. This departs from the
  brief's W&B or MLflow, as W&B needs a login and these files are enough for one person.
- **Mixed precision:** Used on CUDA, off on MPS, where autocast is less reliable.

## D-006: Half precision is exported from PyTorch, not converted afterwards

- **Decision:** Convert the module to half precision and export it, with casts at inputs and outputs.
- **Alternative:** `onnxconverter-common` produced graphs that ONNX Runtime refused to load.
- **Also:** timm's export mode computes "same" padding once. Otherwise run-time padding arithmetic
  enters the graph and slows inference. A test confirms that predictions do not change.

## D-007: Location prior

- **Decision:** A small MLP on sine and cosine of latitude, longitude and day, trained with the SINR
  "assume negative" loss (Cole et al., 2023).
- **Why:** Records show presence only. Treating other species at a site, and all species at a random
  site, as weak absences gives the model negatives. Positive records are weighted up to balance them.
- **Leak prevention:** GBIF republishes iNaturalist. Records matching held-out photos are removed by
  observation ID, and for the geographic split so is every record in a held-out block.
- **Threatened species:** Coordinates are snapped to 0.5 degrees before training, because a public
  model can be queried across a grid to rebuild a map.
- **Floor:** The prior has a lower limit, so that it cannot entirely rule out a genuine vagrant.

## D-008: Calibration, unknown inputs and fallback

- **Temperature scaling:** One parameter, which cannot change the ranking and suits a modest set.
- **Two validation folds:** Fold A fits temperatures. Fold B tunes fusion and thresholds.
- **Second temperature:** Fitted after fusion, because fusion changes confidence.
- **Energy and top probability:** Both are reported, against near and far unknowns separately, since
  near unknowns are much harder. Threshold-tuning species are kept apart from those used to report.

## D-009: Habitat models use NASA NEX-GDDP-CMIP6 and boosted trees

- **Decision:** Monthly CC0 ensemble median, p10 and p90 fields at 0.25 degrees, read as windows over
  HTTP. Bioclimatic variables are derived by one tested function. Models are LightGBM per species.
- **Alternatives:** WorldClim is excluded by the owner's rule. CHELSA has no clear blanket licence in
  what could be checked. Maxent via `elapid` is an equal method, but trees need no feature design.
- **Validity measures:** Spatial block cross validation, target-group background, and a mask for
  climates beyond the training range. Fitting uses the wider Australasian window to avoid truncating
  niches. A random-background comparison on 30 species reports whether target-group mattered.
- **Baseline:** The models' own 1995-2014 runs, so change reflects climate shift, not model bias.
- **Known weakness:** Percentile fields are not physically consistent climates, so agreement is a
  rough guide only. Resolution is about 25 km. The chart uses this same data because Open-Meteo's
  Climate API has one scenario and stops at 2050.

## D-010: Audio features live inside the ONNX graph

- **Decision:** A log mel spectrogram built from fixed convolutions and exported with the model.
- **Alternative:** A TypeScript port needs its own parity test and could drift. ONNX's STFT operator
  may lack WebGPU support. A test confirms the convolutions match `torch.stft`.
- **Recordings:** iNaturalist sounds replace Xeno-canto, which needs an API key. There are fewer per
  species, which limits audio accuracy.

## D-011: Public image storage is off, and the detector needs a separate decision

- **Feedback:** Stores species, confidence and a coarsened place, never a photo. Anonymous public
  image storage carries a real risk of illegal content. Retraining accepts consented images only
  from files the owner supplies.
- **Detector:** `MDV6-mit-yolov9-c` is the chosen detector for its MIT listing and small size. Running
  Microsoft's model code to export it needs the owner's approval. See the README status section.

## D-012: Cache Components are off

- **Decision:** Removed the scaffold's `cacheComponents` and `partialPrefetching` options.
- **Reason:** The app has no use for them, and with them on, unknown species pages returned status
  200 and `dynamicParams` was unavailable.

## D-013: Isolation headers on worker scripts

- **Decision:** `/identify` and `/camera-trap` use `COEP: credentialless` for multithreaded WebAssembly,
  and `/_next/static/*` carries matching headers.
- **Reason:** Without them, the browser refused to start the worker. Only a real browser test showed this.

## D-014: Expanded to 550 species and up to 1,000 photos each

- **Decision:** At the owner's request, grow from 200 species and 300 photos each (about 60,000) to
  550 species and 1,000 photos each (about 386,000). Sound rises to 200 recordings per species.
- **Alternatives:** All 831 species was rejected, because over 250 have fewer than 100 observations,
  leaving too few to test fairly. A higher photo cap adds download and storage with falling returns.
- **Run time:** Each epoch draws at most 150,000 balanced images, which keeps a training run at about
  8 hours on the M4 rather than about 15. Rare species repeat more often, so augmentation matters.
- **Reuse:** Selections are seeded per species and prefer files on disk, so earlier downloads count.
- **Benefit:** A real long tail now exists, and species such as close relatives of common birds are
  classes rather than unknowns.
