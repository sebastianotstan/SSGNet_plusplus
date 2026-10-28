# Dataset and split protocol

This release contains manifests and preparation tools, not source data. Obtain each source dataset from its original distribution and follow its licence and citation requirements. The manifests use dataset-relative paths and file hashes to identify the evaluated image/mask pairs and to check local data integrity.

The internal follow-up test images were partitioned from validation data that had previously been used in the earlier study. This is a post-hoc internal reporting subset, not an untouched prospective test set. The external protocol uses separately assembled cohorts, but its image-level nature and dataset-specific origin limit independence claims. Patient/case identifiers were not available; matching checks are image-content based and cannot establish patient-level independence.

Before training, run the manifest/data verification commands in `src/ssgnet/preflight.py` and the external-data verification utility as applicable. Do not use test scores to choose configurations or tune thresholds. Details and provenance are documented in `docs/EXTERNAL_DATA.md`.
