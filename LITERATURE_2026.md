# GCD Literature Map — checked through 1 October 2026

This is a targeted map for positioning LoRGCD, not a complete survey. Venue labels below were checked against CVF/PMLR/ICLR/NeurIPS records or the cited arXiv page.

| Work | Venue/date | Main idea | Relation to LoRGCD gap | Link |
|---|---|---|---|---|
| Generalized Category Discovery — Vaze et al. | CVPR 2022 | Formalizes GCD; semi-supervised clustering and class-count estimation | Task origin | https://openaccess.thecvf.com/content/CVPR2022/html/Vaze_Generalized_Category_Discovery_CVPR_2022_paper.html |
| SimGCD — Wen et al. | ICCV 2023 | Parametric classifier, self-distillation, entropy regularization | Primary implementation baseline | https://github.com/CVMI-Lab/SimGCD |
| CMS — Choi et al. | CVPR 2024 | Mean-shift integrated into contrastive discovery | Must compare against isotropic mean shift | https://openaccess.thecvf.com/content/CVPR2024/html/Choi_Contrastive_Mean-Shift_Learning_for_Generalized_Category_Discovery_CVPR_2024_paper.html |
| HypCD — Liu et al. | CVPR 2025 | Hyperbolic representation/classifier for hierarchical structure | Global constant-curvature geometry, not a sample-wise metric field | https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.html |
| GET — Wang et al. | CVPR 2025 | CLIP text-embedding synthesis and dual visual/semantic branches | Multimodal semantic transfer | https://openaccess.thecvf.com/content/CVPR2025/html/Wang_GET_Unlocking_the_Multi-modal_Potential_of_CLIP_for_Generalized_Category_CVPR_2025_paper.html |
| APL — Dai et al. | CVPR 2025 | Adaptive part discovery for fine-grained GCD | Local visual parts, complementary to feature-neighborhood geometry | https://openaccess.thecvf.com/content/CVPR2025/html/Dai_Adaptive_Part_Learning_for_Fine-Grained_Generalized_Category_Discovery_A_Plug-and-Play_CVPR_2025_paper.html |
| ConGCD — Tang et al. | ICCV 2025 | Primitive decomposition and multiplex consensus | Structured primitives/consensus, not local tangent metrics | https://openaccess.thecvf.com/content/ICCV2025/html/Tang_Dissecting_Generalized_Category_Discovery_Multiplex_Consensus_under_Self-Deconstruction_ICCV_2025_paper.html |
| Attention Focusing — Xu et al. | ICCV 2025 | Token importance and pruning to suppress background distraction | Improves visual cues rather than neighborhood metric | https://openaccess.thecvf.com/content/ICCV2025/html/Xu_A_Hidden_Stumbling_Block_in_Generalized_Category_Discovery_Distracted_Attention_ICCV_2025_paper.html |
| NC-GCD — Han et al. | NeurIPS 2025 | Fixed ETF prototypes + consistent supervised/unsupervised alignment | Global prototype geometry | https://proceedings.neurips.cc/paper_files/paper/2025/hash/be36e50757bf9cd280aa74f89a7d1c23-Abstract-Conference.html |
| BIA — Tang et al. | ICLR 2026 | Bures/isotropy alignment; nuclear-norm surrogate for token spectrum | Global/batch isotropy; strongest geometry contrast | https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b91cefbfa52340af2f16f249572dc76-Abstract-Conference.html |
| SpectralGCD — Caselli et al. | ICLR 2026 | Cross-modal image-concept representation + spectral concept filtering | “Spectral” is concept filtering, not graph-geodesic class-count estimation | https://proceedings.iclr.cc/paper_files/paper/2026/hash/1ece70d2259b8e9510e2d4ca8754cecf-Abstract-Conference.html |
| EAGC — Zheng et al. | CVPR 2026 | Gradient-direction/subspace coordination | Optimization-level failure mode | https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_The_Devil_Is_in_Gradient_Entanglement_Energy-Aware_Gradient_Coordinator_for_CVPR_2026_paper.html |
| ATCG — Han et al. | CVPR 2026 | Analogical textual concepts for visual-textual GCD | Semantic transfer, strongest fine-grained multimodal comparator | https://openaccess.thecvf.com/content/CVPR2026/html/Han_Learning_Like_Humans_Analogical_Concept_Learning_for_Generalized_Category_Discovery_CVPR_2026_paper.html |
| TAR — Yang et al. | CVPR 2026 | Token-aware refinement to mitigate attention artifacts | Fine-grained token/local cue comparator | https://openaccess.thecvf.com/content/CVPR2026/html/Yang_TAR_Token-Aware_Refinement_for_Fine-grained_Generalized_Category_Discovery_CVPR_2026_paper.html |
| CoGe-GCD — Tang et al. | ICML 2026 | Compositional primitives and structure-preserving calibration | Structured token geometry/compositional generalization | https://proceedings.mlr.press/v306/tang26ai.html |
| CPF-GCD — Dai et al. | ICML 2026 | Low-rank compositional primitive fields | Explicit structured representation prior, not local sample metric | https://proceedings.mlr.press/v306/dai26b.html |
| CURE — Bian et al. | ICML 2026 | Decouples representation learning from conflicting consistency/uniformity | Objective/semantic regularization comparator | https://proceedings.mlr.press/v306/bian26d.html |
| RCA — Yu et al. | ICML 2026 | Evidential confidence alignment across labeled/unlabeled data | Probability-simplex confidence geometry, not feature-manifold geometry | https://proceedings.mlr.press/v306/yu26l.html |
| FourEx — Rastegar et al. | ECCV 2026 / arXiv Aug. 2026 | Low/high-frequency self-supervision for fine-grained GCD | Fine-grained frequency cues | https://arxiv.org/abs/2608.08963 |
| TT-GCD / PACT — Mishra et al. | arXiv 27 Sep. 2026 | Test-time adaptation + known/novel discovery under distribution shift | New deployment setting; useful robustness extension | https://arxiv.org/abs/2609.33937 |

## Positioning statement

The defensible claim is **not** “geometry has not been studied in GCD.” It has. The sharper gap is:

> Existing geometry-oriented GCD work largely imposes a global/homogeneous representation prior (spherical, hyperbolic, isotropic spectrum, fixed prototype arrangement), a compositional token structure, or probability-space/optimization constraints. LoRGCD instead estimates a **sample-wise local SPD metric field plus local tangent orientation**, re-selects graph edges under a tangent-bundle energy, distills that graph into ordinary cosine features, and reuses the same geometry for unknown-category-count estimation.

## Theoretical distinction from global geometry

If one global SPD metric `M` makes two local covariances isotropic up to scale, then `Sigma_a = (c_a/c_b) Sigma_b`. Therefore non-proportional local covariance tensors cannot both be standardized by one global metric. This is the cleanest theoretical motivation for a spatially varying metric field and should be stated explicitly against global Mahalanobis/isotropy-style alternatives.

## Novelty-search terms to rerun immediately before submission

Search titles/abstracts/equations for combinations of: generalized category discovery + local Mahalanobis, local covariance, tangent space, local PCA, Grassmann, principal angles, anisotropic diffusion, tangent bundle, Riemannian graph, manifold graph, metric field, local whitening, geodesic graph, spectral class-number estimation.

The targeted search used to build this repository did not surface the exact combination implemented here, but absence from search results is not proof of novelty.
