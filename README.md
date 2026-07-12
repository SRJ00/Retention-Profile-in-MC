# Singleton and set retention in finite Markov chains

This repository contains the submission source for:

> **From Singleton to Set Retention in KL Contraction of Finite Markov Chains**

The authoritative manuscript is **main-revised-adv.tex**. Earlier development
versions and review artifacts are not part of this repository or the
submission package.

The previous optimizer-based table and correlation study has been removed.
The current manuscript makes no numerical-optimizer claims, and the retired
computations are not evidence for, and must not be cited by, the current
paper.

## Submission files

The manuscript compiles from the following self-contained source set:

- main-revised-adv.tex
- figures/singleton-retention-profiles.pdf
- scripts/generate_retention_profile_figure.py
- scripts/validate_retention_claims.py
- README.md
- requirements-reproducibility.txt

The file **refs.bib** mirrors the embedded bibliography for reuse, but the
manuscript currently uses its internal thebibliography environment and
therefore does not require BibTeX.

## Reproduce the figure and checks

Python 3.10 or later is recommended.

    python -m pip install -r requirements-reproducibility.txt
    python scripts/generate_retention_profile_figure.py
    python scripts/validate_retention_claims.py

The figure generator writes a vector PDF to
**figures/singleton-retention-profiles.pdf**. The validator uses only the
Python standard library and checks:

- positivity, row normalization, stationarity, and detailed balance of the
  multiplicity construction;
- agreement of the matrix calculation with the displayed block-divergence
  formulas;
- the order and inequalities used in the diagonal choice of N_m and
  delta_m;
- the supplementary three-state spectra and profiles;
- the symmetric binary comparison; and
- the uniform set-bottleneck simplification.

The script is a transcription check; the proofs in the manuscript remain
the mathematical justification.

## Build the manuscript

Run pdfLaTeX twice from the repository root:

    pdflatex -interaction=nonstopmode -halt-on-error main-revised-adv.tex
    pdflatex -interaction=nonstopmode -halt-on-error main-revised-adv.tex

No shell escape, external data, or network access is required. A clean build
should contain no undefined references, undefined citations, missing figures,
or overfull boxes.

## Submission-package check

Before submission, extract the source archive into an empty directory and
repeat the two pdfLaTeX commands there. The data-and-code statement in the
paper refers only to files listed above, all of which must be present in the
archive.
