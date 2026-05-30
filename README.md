# Retention Profiles and KL Contraction in Finite Markov Chains

This bundle contains the SSRN-ready manuscript and a minimal reproducibility
package.

## Contents

```
publishable/
  main.tex          # manuscript (inline bibliography; one-shot pdflatex)
  refs.bib          # parallel BibTeX artifact (not used by main.tex)
  README.md         # this file
  reproducibility/
    kl_contraction_audit.py            # master audit script
    simulate_rho_spectral_cheeger.py   # generates correlations cited in §7
    verify_table_remark_7_6.py         # driver for Table in Remark 7.6
    verify_table_remark_7_6_results.json   # cached numerical outputs
    requirements.txt
```

## Building the PDF

```
pdflatex main.tex
pdflatex main.tex
```

The bibliography is inlined via `\begin{thebibliography}`; no BibTeX run is
required. `refs.bib` is shipped as a companion artifact for downstream
journal submissions or third-party citation.

## Reproducing the numerics

From the `reproducibility/` directory:

```
pip install -r requirements.txt
python kl_contraction_audit.py
python verify_table_remark_7_6.py
python simulate_rho_spectral_cheeger.py
```

Random seed is fixed at `20260511`. Reference environment: Python 3.12.13,
NumPy 2.3.5.

- `verify_table_remark_7_6.py` regenerates the table cited in Remark 7.6;
  outputs are written to `verify_table_remark_7_6_results.json`.
- `simulate_rho_spectral_cheeger.py` reproduces the Spearman correlations
  reported in §7 across the 12-family suite (n = 46 chains).
- `kl_contraction_audit.py` is the self-contained audit referenced in the
  Code and Data Availability statement.
