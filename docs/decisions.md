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
