# Submission metadata

## Title

Single-Slot Tomography Cannot Self-Certify Multitime Response Dimension

## Author

Yongzi Chen (ORCID 0009-0008-6202-002X)
Department of Physics and Materials, The Hong Kong Polytechnic University,
Hung Hom, Hong Kong, China
23103646d@connect.polyu.hk

## arXiv

- Primary category: quant-ph
- Cross-list (optional): none
- Comments field: "4 pages (including End Matter) + 21 pages Supplemental Material, 1 figure.
  Code and data: https://github.com/ccyyyYyzz/flat-extension-certificate
  (release v1.0)"

## PhySH subject headings (APS form)

Primary discipline: Quantum Information Science

Concepts, in order of relevance:

1. Quantum tomography
2. Open quantum systems & decoherence
3. Quantum characterization, verification, & validation
4. Quantum control
5. Quantum information processing with superconducting circuits

## 100-word justification (why PRL)

Every quantum platform asks when a device has been characterized completely
enough. We prove this question cannot be answered from within: a minimal
two-qubit collision model whose complete single-time tomography is exactly
blind to a physical multitime response direction, and a delayed-chain family
showing no finite plateau ever certifies closure. We then give the repair - a
rank-saturated flat-extension certificate that converts one externally
justified order bound into an all-depth guarantee with explicit failure
semantics - and stress-test its obstruction and reference-calibration
branches in sealed, preregistered blind tests on two IBM processors,
including a caught-and-corrected device-locked false positive. This reframes
tomographic completeness as a falsifiable protocol property, of immediate
interest across platforms.

## Figure alt-text (accessibility field)

Four-panel figure. Panel (a): response versus memory rotation delta; three
blue horizontal lines at -1, 0, and 1 show single-slot statistics that remain
exactly constant, while one vermillion curve equal to sine of delta rises
through zero with unit slope, marked by a dotted tangent line. Panel (b):
numerical rank of the first-jet Hankel matrix versus tested horizon; the rank
stays at one across a shaded false-plateau region for horizons zero to three,
then jumps to two at horizon four, half the hidden-chain length. Panel (c):
vertical flowchart from a box labeled finite core H through a check box
labeled rank saturation, flatness, and gap, to a green terminal box labeled
all-depth extension; two curved side branches lead to an amber chip labeled
inconclusive and a vermillion chip labeled class excluded. Panel (d):
scatter plot of absolute z-scores normalized by each decision family's frozen
gate for three sealed hardware runs labeled marrakesh, fez, and fez-v3;
vermillion dark-direction points lie well above the dashed unit line, grey
null squares lie below it except two starred fez points just above the line
labeled device floor, and blue QPT control triangles all lie below the line.

## Data availability statement (as in the Letter)

The data and software supporting this study - including IBM job outputs,
calibration snapshots, sealed manifests, verification hashes, analysis code,
and manuscript-generation scripts - are openly available in the public
repository release v1.0 at
https://github.com/ccyyyYyzz/flat-extension-certificate/releases/tag/v1.0
