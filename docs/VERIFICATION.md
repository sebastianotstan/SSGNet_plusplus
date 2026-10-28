# Package verification

Checks run against this packaged directory on 28 October 2026:

- The 17-test suite passed with the local Python environment.
- The CPU fixture smoke run passed: two toy seeds completed R0/R1/R2 training, pseudo-mask updates, separate held-out evaluation, and paired seed analysis. These fixture scores are software checks, not research results.
- Bash syntax checks passed for the included shell launchers.
- Internal and external matrix dry runs each generated 112 PBS command lines; dry-run mode submitted no jobs.
- The model, training, evaluation, refinement, metrics, and data-manifest source files were copied from the locally tested `revision/` tree. The smoke harness was adjusted only to locate the root-level configuration folder in this package layout.

This does not constitute a fresh full research-data run from this reorganized package, a fresh CUDA integration run, or validation on a target HPC system. The site environment template and PBS directives require local configuration and a one-job validation. The original local implementation had been used for the reported local/HPC experiments, but this archive's new path wiring has only been checked by the tests and smoke workflow above.
