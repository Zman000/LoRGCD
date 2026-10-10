# LoRGCD — Tangent-Bundle Geometry for Generalized Category Discovery

Research-grade PyTorch scaffold for a theory-backed Generalized Category Discovery (GCD) submission. The central claim to test is:

> GCD should not force all samples to share one homogeneous geometry. A useful discovery graph should adapt to **sample-wise local covariance** and **local tangent orientation**, then distill that geometry into a simple inference representation.

**Status (1 Oct 2026):** method + training/evaluation/ablation code + theory/manuscript scaffold are complete enough to start experiments. The repository contains **no fabricated benchmark numbers** and does not claim CVPR acceptance or SOTA performance.

## 1. Research positioning

The field has moved well beyond the original GCD baseline. Important directions include SimGCD (parametric self-distillation), CMS (mean shift), HypCD (hyperbolic geometry), Attention Focusing/TAR (local tokens), NC-GCD (ETF prototype geometry), MTMC/BIA (token-manifold capacity/global isotropy), EAGC (gradient conflict), ATCG/CoGe-GCD (semantic/compositional reasoning), FourEx (frequency self-supervision), and the Sep. 2026 TT-GCD setting under test-time distribution shift.

The precise gap used here is **not** “geometry is new to GCD.” It is:

> Most geometry-oriented GCD methods impose a global/homogeneous prior or class/prototype structure. LoRGCD instead constructs a **sample-wise local metric field plus tangent-space orientation**, re-selects graph neighbors under that tangent-bundle energy, and uses the same graph for representation distillation, pseudo-target smoothing, and unknown-$K$ estimation.

See `LITERATURE_2026.md` for the targeted literature map and novelty-search terms.

## 2. Method at a glance

For each normalized semantic feature $z_i$, estimate a local tangent basis $U_i$ and tangent variances $\lambda_i$ from a small cosine-neighbor estimation set. Define the local inverse-covariance metric

$$
M_i=U_i^\top\operatorname{diag}(\lambda_i^{-1})U_i
+\sigma_{\perp,i}^{-2}(I-U_i^\top U_i),
\qquad
\sigma_{\perp,i}^2=\rho\operatorname{mean}(\lambda_i).
$$

The symmetric standardized displacement is

$$
d_M^2(i,j)=\tfrac12(z_j-z_i)^\top(M_i+M_j)(z_j-z_i).
$$

LoRGCD adds an independent tangent-orientation term

$$
d_T^2(i,j)=r-\|U_iU_j^\top\|_F^2
=\sum_{\ell=1}^{r}\sin^2\theta_\ell,
$$

where $\theta_\ell$ are principal angles between local tangent spaces. After robust median normalization,

$$
E_{ij}=d_M^2(i,j)/s_M+d_T^2(i,j)/s_T.
$$

Crucially, cosine kNN is only an **estimation/candidate mechanism**. Final graph neighbors are re-selected under $E_{ij}$. This lets geometry actually change neighborhoods rather than merely reweighting a fixed graph.

The graph drives two losses:

- **RAD — Riemannian Affinity Distillation:** match student cosine-neighbor probabilities to the tangent-bundle affinity.
- **GMD — Geometry-aware Mean-Shift Distillation:** one/few anisotropic graph-diffusion steps produce more stable teacher pseudo-targets.

At inference there is **no SVD, graph, or extra module**; only the trained backbone and classifier remain.

## 3. Why the theory is nontrivial

`THEORY.md` develops six results/arguments aligned with the implementation:

1. **Global-metric impossibility:** one SPD metric can simultaneously isotropize two local covariance tensors only when those tensors are proportional. Heterogeneous neighborhoods therefore require a spatially varying metric field.
2. **Standardized separation:** under a shared local covariance, inverse covariance leaves expected within-class standardized energy at $2d$ but adds $\Delta^\top\Sigma^{-1}\Delta$ to cross-class energy. Low-variance discriminative directions are amplified relative to high-variance nuisance motion.
3. **Tangent intersection:** $r-\|U_iU_j^\top\|_F^2=\sum\sin^2\theta_\ell$, so two manifolds can be arbitrarily close in position while tangent disagreement remains positive.
4. **Local-PCA consistency:** standard smooth-manifold conditions justify the tangent estimator; this classical theorem is cited as justification, not claimed as new.
5. **Natural-gradient mode seeking:** under a frozen local metric, anisotropic mean shift obeys $m(z)-z=h^2M^{-1}\nabla\log \hat p(z)$, i.e. a local natural-gradient density-ascent step.
6. **Spectral class count:** an ideal $K$-component graph has $K$ zero normalized-Laplacian eigenvalues; reduced cross-class graph mass should make the $K$-th eigengap more stable.

These produce falsifiable predictions instead of merely introducing another loss coefficient.

## 4. SimGCD baseline fidelity

The training scaffold was aligned to the official SimGCD recipe rather than inventing a stronger baseline:

- DINO ViT-B/16-style backbone default (`vit_base_patch16_224.dino` through `timm`);
- only the final ViT block trainable by default (`--freeze_blocks 11`);
- 3-layer projection MLP for representation contrastive losses;
- classifier and LoRGCD geometry operate on the normalized backbone semantic feature;
- balanced labeled/unlabeled sampler;
- SGD, momentum 0.9, LR 0.1, weight decay `1e-4`, cosine schedule;
- `sup_weight=0.35`, `memax_weight=2`;
- teacher temperature warmup from 0.07 to 0.04 for 30 epochs;
- final-epoch model is the primary protocol; hidden evaluation labels are **not** used for checkpoint selection.

The data loader is intentionally self-contained, so exact paper reproduction still requires official SSB class splits and matching augmentation/data preprocessing. For an archival comparison, also plug `geometry.py` + RAD/GMD into the official SimGCD repository and confirm the baseline number first.

## 5. Repository

```text
LoRGCD_CVPR/
├── train.py                    # SimGCD-aligned training + LoRGCD
├── evaluate.py                 # final checkpoint evaluation + feature export
├── extract_features.py         # feature extraction entry point
├── models.py                   # DINO/timm backbone, projection head, cosine classifier
├── datasets.py                 # CIFAR, ImageFolder, CSV + GCD labeled masks
├── losses.py                   # baseline losses + RAD
├── geometry.py                 # local metric, Grassmann term, graph, spectral K
├── metrics.py                  # one global Hungarian assignment: All/Old/New + NMI/ARI
├── estimate_k.py               # sparse TBRA eigengap estimator
├── cluster_eval.py             # k-means at known/estimated K
├── geometry_diagnostics.py     # graph purity/cross-class mass/anisotropy diagnostics
├── synthetic_anisotropy.py     # Proposition-1 controlled test
├── synthetic_intersections.py  # Proposition-2 tangent test
├── ablate.py                   # core causal ablations
├── structural_sweeps.py        # k/r/rho/steps/label-fraction sweeps
├── run_cvpr_suite.py           # multi-dataset / multi-seed launcher
├── aggregate_results.py        # final-epoch aggregation by default
├── paper_tables.py             # aggregate evaluation.json into CSV
├── visualize.py                # t-SNE diagnostic
├── configs/cvpr_suite.yaml
├── tests/
│   ├── conftest.py
│   └── test_geometry.py
├── THEORY.md
├── EXPERIMENTS.md
├── LITERATURE_2026.md
├── PAPER_DRAFT.md
└── requirements.txt
```

## 6. Installation

```bash
conda create -n lorgcd python=3.10 -y
conda activate lorgcd
pip install -r requirements.txt
```

Check the DINO model name available in your installed `timm`:

```bash
python - <<'PY'
import timm
print(timm.list_models('*dino*', pretrained=True))
PY
```

If `vit_base_patch16_224.dino` is named differently in your `timm` build, pass the available equivalent with `--backbone`.

## 7. Data protocol

GCD has labels only for a subset of examples from old/known classes. The unlabeled set contains both old and new classes.

The generic benchmark defaults are deterministic: CIFAR-10 uses the first 5/10 class IDs as old, CIFAR-100 uses the first 80/100, and ImageNet-100 uses 50/100. For **fine-grained SSB datasets** (CUB, Cars, Aircraft), do not rely on the fallback: supply the official old-class split:

```bash
--split_file /path/to/old_classes.json
```

Accepted JSON formats:

```json
[0, 2, 7, 9]
```

or

```json
{"old_classes": [0, 2, 7, 9]}
```

Fine-grained datasets can be arranged as `ImageFolder`, or supplied through a CSV with columns `path,label`.

## 8. First baseline check

Before testing LoRGCD, reproduce the baseline with the geometry losses disabled:

```bash
python train.py \
  --dataset cifar100 \
  --data_root /data/cifar \
  --download \
  --lambda_rad 0 \
  --lambda_gmd 0 \
  --amp \
  --output runs/cifar100/simgcd_seed42
```

The point of this run is **baseline fidelity**, not tuning. If your baseline is materially below the official SimGCD result under the same split/backbone protocol, fix the pipeline before judging LoRGCD.

## 9. Full LoRGCD run

```bash
python train.py \
  --dataset cifar100 \
  --data_root /data/cifar \
  --download \
  --geo_k 16 \
  --geo_rank 8 \
  --geo_shrinkage 0.10 \
  --lambda_rad 0.5 \
  --lambda_gmd 0.5 \
  --amp \
  --output runs/cifar100/lorgcd_seed42
```

Primary paper reporting should use `last.pt` unless a validation protocol independent of novel test labels is specified.

## 10. Evaluation

```bash
python evaluate.py \
  --checkpoint runs/cifar100/lorgcd_seed42/last.pt \
  --data_root /data/cifar
```

This writes `evaluation.json` and `features_eval.npz`.

`metrics.py` computes a Hungarian mapping **once on all unlabeled evaluation samples**, then uses that same mapping to report Old and New accuracy. This avoids inflating subset scores by separate re-matching.

## 11. Core ablations

```bash
python ablate.py \
  --dataset cifar100 \
  --data_root /data/cifar \
  --base_output runs/ablations \
  --seeds 42,17,123
```

The launcher includes:

- SimGCD baseline;
- RAD only;
- GMD only;
- full TBRA + RAD + GMD;
- isotropic cosine graph;
- one global Mahalanobis metric;
- local Mahalanobis **without tangent-bundle consistency**;
- full method without known-label graph anchors.

The most important causal contrast is `local_no_tangent_bundle` vs `full`.

## 12. Structural sweeps

Examples:

```bash
python structural_sweeps.py --sweep k --data_root /data/cifar
python structural_sweeps.py --sweep rank --data_root /data/cifar
python structural_sweeps.py --sweep rho --data_root /data/cifar
python structural_sweeps.py --sweep steps --data_root /data/cifar
python structural_sweeps.py --sweep label_fraction --data_root /data/cifar
```

These are theory/robustness experiments, not a license to select a different setting per test dataset.

## 13. Synthetic theory checks

Isolate inverse-covariance behavior:

```bash
python synthetic_anisotropy.py --ratios 1,2,4,8,16
```

Isolate tangent-orientation behavior:

```bash
python synthetic_intersections.py --angles 0.3,0.6,1.0,1.3,1.57
```

In the deterministic smoke test used while building this repo, local whitening only became beneficial at strong anisotropy, while tangent-bundle consistency improved graph purity over local-Mahalanobis-only across all tested tangent angles. These values are development sanity checks, **not benchmark claims**.

## 14. Geometry diagnostics on real features

```bash
python geometry_diagnostics.py \
  --features runs/cifar100/lorgcd_seed42/features_eval.npz \
  --max_points 4096
```

Report graph purity and cross-class mass only as **post-hoc diagnostics**; ground-truth novel labels must never feed training.

## 15. Unknown category count

```bash
python estimate_k.py \
  --features runs/cifar100/lorgcd_seed42/features_eval.npz \
  --min_k 50 \
  --max_k 140 \
  --neighbors 20 \
  --rank 8 \
  --candidate_factor 4 \
  --unlabeled_only
```

The offline estimator first builds a larger cosine candidate pool and then re-ranks it with the same local standardized displacement + tangent-space term used during training. This avoids the inconsistency of estimating $K$ from a fixed cosine graph.

Then evaluate clustering at the estimated $K$:

```bash
python cluster_eval.py \
  --features runs/cifar100/lorgcd_seed42/features_eval.npz \
  --k <K_HAT>
```

Known-$K$ should remain the primary table for fair comparison; unknown-$K$ is a separate robustness claim.

## 16. Multi-dataset suite

Edit paths in `configs/cvpr_suite.yaml`, then run:

```bash
python run_cvpr_suite.py --config configs/cvpr_suite.yaml
```

Evaluate each `last.pt`, then aggregate:

```bash
python aggregate_results.py --root runs/cvpr_suite --reduction last
python paper_tables.py --root runs/cvpr_suite --output paper_results.csv
```

`aggregate_results.py --reduction best` exists only for debugging and should not be used for the paper unless a legitimate validation split defines model selection.

## 17. What must be shown for a credible CVPR submission

A strong result is not “+0.7 after tuning.” The paper should establish a causal chain:

1. baseline features exhibit local covariance anisotropy and/or tangent disagreement;
2. TBRA reduces cross-class graph mass relative to cosine/global/local-Mahalanobis controls;
3. RAD transfers that improvement into the learned semantic feature;
4. GMD improves pseudo-target stability rather than merely oversmoothing;
5. gains concentrate where the theory predicts (high anisotropy / tangent disagreement);
6. fine-grained datasets benefit strongly;
7. the same graph improves unknown-$K$ estimation;
8. the method remains competitive after reporting SVD/tangent-comparison overhead;
9. all main results use at least 3 seeds and fixed hyperparameter protocol;
10. failure cases are reported.

`EXPERIMENTS.md` expands these into tables, figures, reviewer attacks, and robustness studies.

## 18. Validation performed on this code package

The following checks were run in the build environment:

```text
python -m py_compile *.py tests/*.py     PASS
pytest -q tests/test_geometry.py         5 passed
synthetic_anisotropy.py                  PASS
synthetic_intersections.py               PASS
1-epoch FakeData + torchvision:resnet18   PASS
evaluate.py + estimate_k.py               PASS
```

The build environment did not have `timm` installed, so a full DINO training job was **not** executed here. Install `requirements.txt` and first run a short 1–2 epoch CIFAR smoke job before launching full experiments.

## 19. Novelty and claim discipline

A targeted literature search through **1 October 2026** did not surface a GCD method with the exact combination of:

- sample-wise inverse-covariance metric field;
- Grassmann tangent-space agreement;
- neighbor re-selection under the combined tangent-bundle energy;
- affinity distillation into ordinary cosine features;
- geometry-aware pseudo-target diffusion;
- the same geometric graph for spectral category-count estimation.

That makes this a plausible research direction, not a novelty guarantee. Before submission, rerun the search terms in `LITERATURE_2026.md` and compare equations/algorithms, especially any 2026–2027 arXiv/OpenReview work appearing after this repository was generated.
