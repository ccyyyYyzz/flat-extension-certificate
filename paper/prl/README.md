# PRL manuscript source

Title (post-review restructure, R9):

> **Single-Slot Tomography Cannot Self-Certify Multitime Response Dimension**

The Letter's claim chain (proven package: `research/FORMAL_MASTER_THEOREM_AUDIT.md`,
GitHub issues #5/#6, external-brain rounds R1-R10):

1. an exact two-qubit collision model in which every single-slot causal-break
   statistic is constant in a smooth, non-gauge memory rotation that a retained
   two-slot correlator detects at first order (Proposition 1, exact blindness);
2. a delayed-chain family showing that an arbitrarily clean finite Hankel
   plateau certifies nothing without an external order bound;
3. the saturated first-jet flat-extension theorem: given a declared response
   order n_J, a rank-saturated flat core determines every registered value and
   first derivative at arbitrary depth, up to simultaneous similarity;
4. a failure-aware perturbation theorem (truncated-SVD core, two-matrix error
   sum, explicit CERTIFIED / REJECTED / INCONCLUSIVE semantics) and a resource
   count (O(m n_J^2) certificate entries versus the Theta(d^{4k}) affine
   dimension of unrestricted depth-k comb tomography; no universal shot lower
   bound is claimed);
5. sealed blind benchmarks on two IBM processors: ibm_marrakesh clean,
   ibm_fez exposing a device-locked null floor under the absolute gate, and a
   preregistered device-referenced differential rerun (protocol v3) restoring
   full specificity.

## Build

REVTeX 4.2 (TeX Live 2024+):

```bash
cd paper/prl
latexmk -pdf main.tex          # Letter (body + End Matter, 4 pp)
latexmk -pdf supplement.tex    # 21-page Supplemental Material (xr-hyper reads main.aux)
```

The single Letter figure is generated deterministically (hardware numbers are
read from the sealed run JSONs where available; the script self-checks the
family gates against the stored verdicts and enforces a text-overlap gate):

```bash
python figures/make_c7_flat_extension.py   # -> figures/c7_flat_extension.{pdf,png}
```

## arXiv package

`arxiv/ms.tex` is the merged single-document source (Letter + End Matter +
Supplemental Material, one reference list at the Letter position, S-prefixed
counters after the seam). `arxiv/arxiv_submission.tar.gz` contains exactly
`ms.tex`, `ms.bbl`, and `figures/c7_flat_extension.pdf` and compiles
standalone with two pdflatex passes (25 pages, no bibtex needed). Regenerate
the merge after any edit to `main.tex` or `supplement.tex` and recompile
before re-uploading.

## Reproduce the numerics

```bash
PYTHONPATH=src python examples/run_c7_prototype.py               # Proposition 1 certificate numbers
PYTHONPATH=src python examples/run_c7_jet_checks.py              # jet/flat-extension checks
PYTHONPATH=src python experiments/run_step1_separator_power.py   # shot budgets
PYTHONPATH=src python experiments/run_blind_benchmark_dryrun.py  # blind benchmark dry run
```

Sealed hardware artifacts (answer keys, manifests, hashes, scored verdicts)
live under `research/results/` and in the public release:
https://github.com/ccyyyYyzz/flat-extension-certificate/releases/tag/v1.0

## Submission documents

- `cover_letter.md` — signed PRL cover letter
- `SUBMISSION_CHECKLIST.md` — current state of every submission requirement
- `SUBMISSION_METADATA.md` — PhySH terms, figure alt-text, 100-word justification
- `REVIEWER_RISK_REGISTER.md` — anticipated objections and scoped responses
