# PRL submission checklist

## Manuscript

- [x] Title scoped to the proven claim (R9 retitle): single-slot tomography,
      multitime response dimension, no ontic language.
- [x] Letter compiles to 4 pages in `reprint` format: a ~3-page body within
      the PRL length limit, references, and a ~1-page End Matter (excluded
      from the limit by the 2024 policy); Supplemental Material is 21 pages.
- [x] Abstract states the counterexample, the conditional certificate, the
      INCONCLUSIVE semantics, and the two-processor hardware outcome.
- [x] Author identity: Yongzi Chen, Department of Physics and Materials,
      The Hong Kong Polytechnic University (name verified against the official
      PolyU site after the rename from Applied Physics).
- [x] Email footnote on Letter and Supplement; ORCID 0009-0008-6202-002X
      linked via `orcidlink` on both author lines.
- [x] Acknowledgments: IBM Quantum services disclaimer only; no placeholders.
- [x] Data-availability paragraph points at the public artifact release v1.0.

## Theory (all proven and twice independently audited)

- [x] Proposition 1: exact single-slot blindness of the CZ-Rx(delta)-CZ comb;
      slot-2 identity pinning; retained separator <Y_S1 X_S2> = sin(delta).
- [x] Delayed-chain necessity: no fixed number of flat steps certifies closure
      (N > 2h + 2 family).
- [x] Saturated first-jet flat extension: rank-saturated flat core determines
      all-depth registered responses up to simultaneous similarity.
- [x] Perturbation theorem: truncated-SVD core, two-matrix error sum
      eta_cert = eta_obs + b_A + b_F, INCONCLUSIVE on unresolved evidence.
- [x] Resource count O(m n_J^2) vs Theta(d^{4k}); universal shot lower bound
      explicitly withdrawn (R10 repair).
- [x] Honesty ledger O1-O8 in the Supplement enumerates every open assumption.
- [x] Two independent mathematical audits (R9, R10) integrated; external-brain
      rounds R1-R10 archived in the repository issues.
- [x] QPI (Bennink-Lougovski, NJP 21, 083013 (2019)) cited and positioned:
      block-Hankel identification of a fixed repeated channel versus this
      Letter's word-indexed first-jet response series with intermediate
      interventions (PRL-ification round, 2026-07-22).
- [x] Order-conditioned closure theorem stated formally in the body with a
      four-step proof skeleton; index-level machinery moved to End Matter
      (Appendix A: core and proof summary; Appendix B: finite-error verdicts
      and resources; equations A1-A3, B1-B5, outside the length limit).

## Hardware evidence (sealed, preregistered)

- [x] ibm_marrakesh: 3/3 planted darks, 0/5 null false positives, QPT 0/3.
- [x] ibm_fez v2: 5/5 darks, but 2/3 nulls crossed the absolute gate.
- [x] Sealed six-null confirmation: reproducible negative YX floor
      (pooled z = -6.20), consistent with device-locked readout bias.
- [x] Protocol v3 (device-referenced differential, preregistered):
      5/5 darks, 0/3 nulls on the affected processor; floor sign drift
      (-6.20 -> +4.77) between calibration windows documented.
- [x] Answer-key SHA-256 hashes published before unsealing; estimators frozen
      before every scored run.

## Figure (single four-panel figure)

- [x] (a) exact blindness; (b) Hankel-rank false plateau with jump at h = N/2;
      (c) certification pipeline with failure branches (green terminal =
      exact all-depth extension, matching the theorem's scope); (d)
      family-normalized |z| / z*_family with per-family frozen gates and
      in-script gate consistency assertion. Panel order follows the logic
      chain: obstruction -> no-go -> certificate -> hardware stress test.
- [x] Automated text-overlap gate passes; every panel personally inspected at
      pixel zoom after the final edit round.
- [x] Caption states 0/16 QPT detections (moved out of the legend), the v3
      provenance, and the panel-(d) check conditions.

## Reproducibility

- [x] Deterministic figure script reads sealed run JSONs; test suite green
      (153 tests, hardware tests under the dedicated qiskit venv).
- [x] Public artifact repository flat-extension-certificate, release v1.0,
      sole contributor = author.

## arXiv (next action: author uploads)

- [x] Merged single-document source `arxiv/ms.tex` (25 pages) with references
      at the Letter position and S-prefixed SM counters.
- [x] Upload package `arxiv/arxiv_submission.tar.gz` (ms.tex + ms.bbl +
      figure PDF) verified by clean-room compile: two pdflatex passes,
      no bibtex, zero undefined references.
- [ ] Upload to arXiv quant-ph (author account; endorsement may be requested
      for a first quant-ph submission - use the PolyU address).
- [ ] Record the arXiv identifier for the APS submission form.

## APS submission form (after arXiv)

- [x] Cover letter signed (`cover_letter.md`).
- [x] PhySH terms, figure alt-text, and 100-word justification prepared
      (`SUBMISSION_METADATA.md`).
- [ ] Enter ORCID 0009-0008-6202-002X in the APS author form.
- [ ] Attach or reference the arXiv posting; confirm no dual submission.
