# Experimental Plan for a CVPR-Strength LoRGCD Submission

No main-table number is pre-filled. The repository is intended to produce the evidence.

## A. Central hypothesis

Does **sample-wise tangent-bundle geometry** improve GCD beyond homogeneous cosine/spherical geometry, one global Mahalanobis geometry, global hyperbolic/isotropic/prototype priors, and strong representation/optimization baselines?

## B. Benchmark suite

| Type | Dataset | Primary purpose |
|---|---|---|
| generic | CIFAR-100 | rapid ablations and reproducibility |
| generic | ImageNet-100 | stronger pretrained-feature benchmark |
| fine-grained | CUB-200-2011 | pose/part nuisance + subtle species differences |
| fine-grained | Stanford Cars | fine inter-class appearance geometry |
| fine-grained | FGVC-Aircraft | fine shape variation |
| optional long-tail | Herbarium19 | density/imbalance stress test |

Use the official GCD/SSB old/new splits and the same label proportion as the compared baseline.

## C. Baselines

At minimum reproduce or cite/re-run under a matched backbone where code permits:

1. GCD (CVPR 2022).
2. SimGCD (ICCV 2023).
3. CMS (CVPR 2024).
4. HypCD (CVPR 2025).
5. BIA/MTMC family (2025–ICLR 2026 geometry/manifold capacity).
6. EAGC (CVPR 2026) for optimization conflict.
7. SpectralGCD (ICLR 2026) or ATCG (CVPR 2026) for multimodal semantics.
8. CPF-GCD / CoGe-GCD (ICML 2026) for compositional structure.
9. TAR or FourEx for fine-grained local/frequency cues.
10. CURE or RCA (ICML 2026) for consistency/confidence controls.

The cleanest causal result is to run LoRGCD as a module on **the same SimGCD backbone/head and data split**. A second integration into a stronger 2026 baseline would demonstrate complementarity.

## D. Main table

Mean ± sample standard deviation over at least 3 seeds.

| Method | CIFAR100 All/Old/New | ImageNet100 | CUB | Cars | Aircraft |
|---|---:|---:|---:|---:|---:|
| SimGCD | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |
| CMS | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |
| HypCD | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |
| BIA/MTMC | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |
| Strong 2026 baseline | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |
| **LoRGCD** | TO RUN | TO RUN | TO RUN | TO RUN | TO RUN |

Also report NMI/ARI in supplementary tables.

## E. Causal ablation table

| Variant | Local covariance | Tangent term | RAD | GMD | Known anchors | All | Old | New |
|---|---|---|---|---|---|---:|---:|---:|
| SimGCD baseline | no | no | no | no | no | TO RUN | | |
| isotropic graph | no | no | yes | yes | yes | TO RUN | | |
| global Mahalanobis | global | no | yes | yes | yes | TO RUN | | |
| local Mahalanobis | local | no | yes | yes | yes | TO RUN | | |
| TBRA + RAD only | local | yes | yes | no | yes | TO RUN | | |
| TBRA + GMD only | local | yes | no | yes | yes | TO RUN | | |
| full LoRGCD | local | yes | yes | yes | yes | TO RUN | | |
| full, no anchors | local | yes | yes | yes | no | TO RUN | | |

The reviewer-critical contrast is **local Mahalanobis vs full TBRA**. It tests Proposition 2 independently of inverse-covariance reweighting.

## F. Theory diagnostics

### F0. Global-metric impossibility diagnostic

Estimate local covariance tensors on the frozen baseline representation. For pairs of neighborhoods, measure normalized covariance disagreement after optimal scalar rescaling,

\[
\delta(\Sigma_i,\Sigma_j)=\min_{c>0}\frac{\|\Sigma_i-c\Sigma_j\|_F}{\|\Sigma_i\|_F}.
\]

If these residuals are broadly nonzero, the data empirically violate the condition under which one global whitening metric could standardize all neighborhoods. Report this separately for coarse- and fine-grained datasets.

### F1. Local anisotropy

For each sample, estimate local tangent variance spectrum and condition/anisotropy statistics. Plot the distribution for SimGCD and LoRGCD features.

Prediction: the method need not make all neighborhoods isotropic; it should learn a representation where semantic neighbors are better connected despite heterogeneous local spectra.

### F2. Gain conditioned on anisotropy

Bin evaluation samples by pre-LoRGCD local anisotropy. Plot change in correctness, neighbor purity, and pseudo-target agreement. Prediction: gain increases with anisotropy until tangent estimation becomes unreliable.

### F3. Tangent disagreement

For cosine-neighbor pairs, compute

\[
d_T^2=r-\|U_iU_j^\top\|_F^2.
\]

Compare its distribution for same-class vs cross-class edges using labels only post hoc. Then measure whether the full TBRA ablation removes high-tangent-disagreement cross-class edges.

### F4. Graph purity and cross-class mass

For evaluation-only labels:

\[
\operatorname{purity}_i=\sum_jA_{ij}\mathbf 1[y_i=y_j].
\]

Report average purity and \(1-\)purity for cosine, global Mahalanobis, local Mahalanobis, and TBRA.

### F5. Pseudo-target stability

Track agreement of teacher pseudo-labels across augmentations and across adjacent epochs. Compare ordinary teacher targets with GMD targets, especially on low-margin samples.

## G. Synthetic theorem validation

### G1. Anisotropic shared-tangent manifolds

Run `synthetic_anisotropy.py`. Sweep \(\sigma_t^2/\sigma_\perp^2\). Compare cosine graph with **local Mahalanobis without tangent term** to isolate Proposition 1.

Expected qualitative pattern: little/no advantage when geometry is close to isotropic; increasing advantage when nuisance tangent variance dominates the low-variance discriminative direction.

### G2. Near-intersecting manifolds

Run `synthetic_intersections.py`. Keep point separation small while varying tangent angle. Compare local-Mahalanobis-only with TBRA.

Expected qualitative pattern: tangent term reduces cross-manifold edges as principal-angle disagreement becomes informative.

### G3. Negative-control synthetic

Construct classes with the same covariance and same tangent orientation but overlapping means. Prediction: TBRA should **not** magically solve the problem. Including this control strengthens the theory section.

## H. Structural robustness (not per-dataset tuning)

| Quantity | Values |
|---|---|
| neighborhood \(k\) | 8, 16, 32, 64 |
| tangent rank \(r\) | 2, 4, 8, 16 |
| normal shrinkage \(\rho\) | 0.03, 0.1, 0.3, 1.0 |
| diffusion steps | 1, 2, 3 |
| labeled old-class fraction | 10%, 25%, 50% |
| backbone | DINO ViT-B/16; DINOv2 equivalent; optional stronger foundation encoder |

Pick one default protocol before the final benchmark sweep. Report the full sensitivity curves.

## I. Unknown-K experiments

Known-\(K\) remains the primary fair comparison. For unknown \(K\), report:

- absolute error \(|\hat K-K|\);
- relative error \(|\hat K-K|/K\);
- downstream All/Old/New at \(\hat K\);
- normalized-Laplacian eigenvalue plots;
- stability across seeds and candidate-pool sizes.

Use the **same eigengap estimator** on cosine, global Mahalanobis, local Mahalanobis, and TBRA graphs. This isolates graph quality from estimator choice.

## J. Efficiency

Report training seconds/epoch, peak GPU memory, percentage overhead over SimGCD, extra trainable parameters, and inference images/s. LoRGCD should add zero inference parameters and no inference-time graph construction.

Separately profile:

- batched local SVD;
- pairwise local metric calculation;
- Grassmann tangent comparison;
- graph construction/RAD/GMD.

If overhead is excessive, test lower-dimensional geometry features or geometry every \(m\) steps as **efficiency ablations**, not silent implementation changes.

## K. Statistical protocol

Use seeds 42, 17, 123 at minimum. Report mean ± sample std. Do not choose the best epoch using hidden unlabeled labels. Use last epoch or a legitimate validation protocol. For the main comparison, a paired bootstrap over evaluation samples can supplement seed-level variation, but do not present pseudo-replication as independent training runs.

## L. Paper figures

1. Method figure: two views → semantic feature → local SVD → standardized displacement + tangent Grassmann term → TBRA → RAD/GMD.
2. Synthetic anisotropy plot.
3. Near-intersection tangent-angle plot.
4. Real-data graph purity / cross-class mass.
5. Gain vs local anisotropy and tangent mismatch.
6. Unknown-\(K\) eigenspectrum.
7. Efficiency/accuracy trade-off.
8. Optional UMAP/t-SNE qualitative visualization (diagnostic only).

## M. Reviewer attack checklist

**“This is just Mahalanobis metric learning.”** Compare global Mahalanobis, local Mahalanobis, and full tangent-bundle affinity with all other components fixed.

**“Grassmann distance is classical.”** Agree; the novelty claim is the GCD-specific tangent-bundle graph, geometric re-selection, distillation, and unified unknown-\(K\) use—not invention of principal angles.

**“The initial kNN is already wrong.”** Measure initial cosine-neighbor contamination; report failure regimes. Full-dataset inference uses a larger candidate pool before TBRA re-ranking.

**“Mean shift explains everything.”** Compare GMD under isotropic, local-Mahalanobis, and TBRA graphs; compare against CMS where protocol permits.

**“Known labels leak into novel discovery.”** Run `--disable_supervised_anchors`; novel labels are never consumed by training.

**“You tuned local geometry per dataset.”** Fix one default setting before the main suite and report structural sweeps separately.

**“Your baseline is weaker/different.”** First reproduce the official SimGCD last-epoch baseline under the same SSB split; then add only the LoRGCD module.

**“Unknown-K improvement is a different estimator.”** Apply one identical spectral estimator to all graph types.

**“The method only works on hand-crafted synthetic data.”** Synthetic experiments validate mechanisms; the main claim stands or falls on broad real benchmarks, especially CUB/Cars/Aircraft.
