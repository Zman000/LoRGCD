# LoRGCD: Tangent-Bundle Geometry for Generalized Category Discovery

> CVPR manuscript scaffold. Every empirical result is intentionally marked **TO RUN**. Do not replace placeholders with invented values.

## Abstract

Generalized Category Discovery (GCD) learns from labeled known classes while categorizing an unlabeled mixture of known and novel categories. Recent methods improve global representation geometry, hierarchical embeddings, prototype structure, local tokens, multimodal semantics, or optimization, but discovery is still typically driven by a homogeneous similarity rule. We study a different failure mode: **local geometric heterogeneity**. In pretrained feature spaces, nuisance variation can produce highly anisotropic class manifolds, while fine-grained class boundaries may lie along locally low-variance directions; additionally, nearby semantic manifolds may have different tangent orientations. We propose **LoRGCD**, which estimates a sample-wise tangent basis and local inverse-covariance metric, then constructs a **Tangent-Bundle Riemannian Affinity (TBRA)** combining standardized displacement with Grassmann tangent-space disagreement. Unlike methods that merely reweight fixed cosine neighbors, LoRGCD re-selects graph neighbors under this geometry. We distill the graph into a conventional cosine embedding using **Riemannian Affinity Distillation (RAD)** and use it to stabilize pseudo-targets through **Geometry-aware Mean-Shift Distillation (GMD)**. The same graph provides a spectral estimator when the number of categories is unknown. We derive standardized separation and tangent-intersection properties that lead to falsifiable predictions about anisotropy and tangent disagreement. LoRGCD adds no inference parameters. **Results: TO RUN.**

## 1. Introduction

GCD asks a model to recognize labeled known categories and simultaneously discover unlabeled novel categories. The task has moved from non-parametric semi-supervised clustering to strong parametric self-distillation, mean-shift objectives, local-token modeling, hyperbolic embeddings, neural-collapse prototypes, manifold-capacity/isotropy regularization, gradient coordination, compositional representations, multimodal reasoning, and online/test-time variants.

These directions reveal that geometry matters, but they leave an important scale under-modeled. A single spherical metric treats every local direction identically. Constant-curvature hyperbolic geometry is globally structured but homogeneous. A global covariance/isotropy objective controls a dataset- or batch-level spectrum. Fixed ETF prototypes regularize class-center angles. None of these mechanisms directly says that **the metric around sample \(i\) can differ from the metric around sample \(j\)** and that the **orientation of their tangent spaces itself contains category evidence**.

Consider two fine-grained classes. Pose and viewpoint may create large within-class motion along a local tangent, whereas the semantic difference between species/model variants may be a smaller displacement orthogonal to that nuisance direction. Cosine distance can rank the nuisance motion as larger. Worse, two manifolds can come close or intersect in ambient feature space; point distance becomes ambiguous, while their tangent orientations remain distinct.

LoRGCD models both effects. Around each feature we estimate a local tangent subspace and its variances by SVD. The inverse local covariance makes high-variance tangent motion cheap and locally thin motion expensive. A second Grassmann term measures disagreement between neighboring tangent spaces. Both terms are median-normalized and combined without a learned mixing coefficient. Cosine neighbors are used only to estimate candidate local structure; final discovery neighbors are re-selected under the tangent-bundle energy.

The graph is a detached teacher. RAD trains ordinary cosine similarities to reproduce its affinities, preserving low-cost inference. GMD uses the same graph for anisotropic pseudo-target smoothing. Finally, spectral analysis of that graph estimates the number of categories when \(K\) is unknown.

### Contributions

1. We formulate **heterogeneous local tangent-bundle geometry** as a GCD failure mode distinct from global curvature, global isotropy, fixed prototype geometry, token selection, and gradient conflict.
2. We introduce **TBRA**, a non-parametric affinity combining local inverse-covariance displacement and Grassmann tangent disagreement, followed by geometric neighbor re-selection.
3. We introduce **RAD** and **GMD**, which respectively distill TBRA into a standard embedding and use it to stabilize pseudo-targets; the geometry adds no inference parameters.
4. We provide analysis showing (i) a single global SPD metric cannot isotropize non-proportional local covariance fields, (ii) inverse covariance increases standardized margin in locally low-variance discriminative directions, (iii) tangent disagreement remains informative at intersecting/near-touching manifolds, and (iv) the positional mean-shift update is a local natural-gradient density-ascent step.
5. We use the same graph for unknown-category-count estimation and define diagnostic experiments that directly test graph purity, anisotropy dependence, tangent disagreement, and eigengap stability. **Empirical claims pending execution.**

## 2. Related Work

### 2.1 Generalized category discovery

Vaze et al. formalized GCD as categorizing an unlabeled mixture of known and novel classes with partial labels only for known categories. SimGCD established a strong parametric baseline based on self-distillation and entropy regularization. Subsequent methods improve clustering dynamics, use unlabeled known-class data more effectively, address local attention/parts, or transfer semantic/compositional information.

### 2.2 Geometry in GCD

HypCD uses hyperbolic space to represent hierarchical structure. NC-GCD fixes ETF prototypes to enforce neural-collapse-style angular structure. MTMC/BIA address global token-manifold capacity and spectrum/isotropy. These works establish geometry as a core issue but primarily impose global or prototype-level constraints. LoRGCD instead estimates a location-dependent metric and tangent orientation.

### 2.3 Local and fine-grained structure

Recent fine-grained GCD methods exploit parts, token refinement, attention focusing, frequency components, textual concepts, or compositional primitives. LoRGCD is complementary: it does not specify what semantic cue a token should encode; it changes how local feature neighborhoods are constructed once features exist.

### 2.4 Manifold and subspace geometry

Local PCA is a classical tangent estimator for smooth sampled manifolds. Mahalanobis distance standardizes anisotropic covariance. Principal angles define canonical distances between subspaces on a Grassmann manifold. LoRGCD does not claim these mathematical objects individually; its contribution is their category-discovery coupling, graph re-selection, representation distillation, and unified unknown-\(K\) use.

## 3. Method

### 3.1 Baseline

Two augmentations \(x_i,x_i'\) yield normalized projections \(z_i,z_i'\). A cosine classifier predicts \(K\) categories. The baseline uses instance contrastive learning, supervised contrastive learning on labeled samples, supervised CE, symmetric two-view self-distillation, and batch-marginal entropy maximization.

### 3.2 Local standardized displacement

Using a cosine estimation neighborhood \(\mathcal N_i\), compute the rank-\(r\) SVD of centered offsets and obtain tangent basis \(U_i\) and tangent variances \(\lambda_i\). Let

\[
\sigma_{\perp,i}^2=\rho\operatorname{mean}(\lambda_i).
\]

Define

\[
M_i=U_i^\top\operatorname{diag}(\lambda_i^{-1})U_i+
\sigma_{\perp,i}^{-2}(I-U_i^\top U_i),
\]

and

\[
d_M^2(i,j)=\tfrac12\delta_{ij}^\top(M_i+M_j)\delta_{ij}.
\]

### 3.3 Tangent-space consistency

For orthonormal rank-\(r\) bases, define

\[
d_T^2(i,j)=r-\|U_iU_j^\top\|_F^2
=\sum_{\ell=1}^r\sin^2\theta_\ell.
\]

This term is zero for aligned local tangents and positive for different tangent orientations, including when point distance is small.

### 3.4 Tangent-Bundle Riemannian Affinity

Normalize the positional and tangent energies by their robust local medians:

\[
E_{ij}=d_M^2(i,j)/s_M+d_T^2(i,j)/s_T.
\]

No additional mixing hyperparameter is introduced. Final graph neighbors are selected by the lowest \(E_{ij}\), not frozen to the original cosine kNN. Affinities are

\[
A_{ij}\propto\exp(-E_{ij}/\tau_g).
\]

Known-labeled same-class edges can be strengthened and different-known-class edges removed. Novel labels are never used.

### 3.5 Riemannian Affinity Distillation

Let \(S_{ij}\) be the trainable cosine-neighbor softmax. Minimize

\[
\mathcal L_{RAD}=-\sum_{ij}A_{ij}\log S_{ij}.
\]

This transfers the non-parametric tangent-bundle neighborhood into the ordinary representation.

### 3.6 Geometry-aware Mean-Shift Distillation

One graph-diffusion step is

\[
\tilde z_i=\operatorname{norm}(\sum_j A_{ij}z_j).
\]

A detached teacher classifier produces pseudo-targets from \(\tilde z_i\), and the opposite augmentation predicts them. This is an anisotropic graph mean-shift approximation, not a claim of exact continuous geodesic integration.

### 3.7 Objective and inference

\[
\mathcal L=\mathcal L_{base}
+\lambda_{RAD}\mathcal L_{RAD}
+\lambda_{GMD}\mathcal L_{GMD}.
\]

The SVD, metric, tangent comparison, and graph are detached. At inference, LoRGCD uses exactly the trained backbone/projector/classifier with no graph or SVD.

### 3.8 Unknown category count

For full-dataset features, construct a sparse TBRA graph using a \(4k\) cosine candidate pool followed by tangent-bundle re-ranking. The normalized-Laplacian eigengap in a plausible interval \([K_{old},K_{max}]\) estimates total \(K\).

## 4. Analysis

### 4.1 Why a local metric field is necessary

If one global SPD metric \(M\) makes two local covariances isotropic up to scale, then \(M^{1/2}\Sigma_iM^{1/2}=c_iI\) implies \(\Sigma_i=c_iM^{-1}\). Thus \(\Sigma_a\) and \(\Sigma_b\) must be proportional. Real feature neighborhoods with different tangent orientations/eigenspectra violate this condition, motivating a sample-wise metric field.

### 4.2 Standardized class separation

For shared covariance \(\Sigma\), the expected within-class inverse-covariance energy equals \(2d\); between classes it is \(2d+\Delta^\top\Sigma^{-1}\Delta\). Thus a low-variance semantic displacement is amplified relative to high-variance nuisance motion.

### 4.3 Intersecting manifolds

The Grassmann chordal term equals \(\sum_\ell\sin^2\theta_\ell\). Hence it can remain positive when two different manifolds approach the same ambient point but possess different tangents. This is information unavailable to cosine distance or a purely positional Mahalanobis metric.

### 4.4 Mean shift as local natural-gradient mode seeking

For a frozen local metric \(M_i\), the anisotropic Gaussian KDE obeys

\[
m_i(z)-z=h^2M_i^{-1}\nabla_z\log\hat p_i(z),
\]

so the positional component of GMD is a Riemannian/natural-gradient ascent step toward local density modes. The tangent-bundle term further removes edges whose local orientations disagree.

### 4.5 Spectral category count

An ideal graph with \(K\) connected category components has \(K\) zero eigenvalues of its normalized Laplacian. Perturbation bounds motivate the prediction that reducing cross-class edge mass stabilizes the \(K\)-th eigengap.

## 5. Experiments

### 5.1 Benchmarks and protocol

Use official GCD/SSB splits on CIFAR-100, ImageNet-100, CUB-200-2011, Stanford Cars, and FGVC-Aircraft; optionally Herbarium19 for imbalance. Use at least three seeds. The primary clustering metric is All/Old/New accuracy with **one Hungarian mapping computed over all unlabeled evaluation samples**, plus NMI/ARI. Known-\(K\) comparison is primary; unknown-\(K\) is a separate robustness setting.

### 5.2 Main comparison

**Table 1 — TO RUN.** Compare GCD, SimGCD, CMS, HypCD, BIA/MTMC, one strong 2026 fine-grained/optimization baseline, one compositional baseline, and LoRGCD under matched backbones where feasible.

### 5.3 Causal ablations

**Table 2 — TO RUN.** Baseline; isotropic cosine graph; global Mahalanobis; local Mahalanobis without tangent term; full TBRA; RAD-only; GMD-only; no labeled anchors.

The most important comparison is **local Mahalanobis vs full TBRA**: it tests whether tangent orientation provides information beyond local covariance scaling.

### 5.4 Geometry diagnostics

**Figure 3 — TO RUN.** Local anisotropy distribution; tangent disagreement distribution for correct/incorrect cosine neighbors; graph purity; cross-class affinity mass; improvement conditioned on anisotropy and tangent mismatch.

### 5.5 Synthetic theory tests

**Figure 4a — TO RUN.** Shared-tangent anisotropic classes: sweep \(\sigma_t^2/\sigma_\perp^2\); local standardized distance should increasingly outperform cosine as Proposition 1 predicts.

**Figure 4b — TO RUN.** Near-intersecting linear/curved manifolds with matched point distances but different tangent angles; TBRA should outperform local-Mahalanobis-only as Proposition 2 predicts.

### 5.6 Unknown-K

**Table 3/Figure 5 — TO RUN.** Compare class-count MAE, relative error, eigenspectra, and downstream clustering for cosine, global Mahalanobis, local Mahalanobis, and TBRA graphs.

### 5.7 Robustness and efficiency

**Table 4 — TO RUN.** Sweep neighborhood size, tangent rank, normal shrinkage, graph-diffusion steps, label fraction, class-count mismatch, and backbone. Report training seconds/epoch, peak memory, parameter count, and inference throughput.

## 6. Limitations

Local tangent estimation requires sufficiently pure and dense neighborhoods. Cross-class contamination can corrupt the SVD before TBRA has a chance to re-rank candidates. Very small batches make local geometry noisy. If two categories overlap and share both covariance and tangent orientation, the proposed geometry has no principled advantage. Strongly varying intrinsic dimension is handled only indirectly by a fixed rank in the current implementation. Batched SVD and all-pairs tangent comparison add training overhead. These are empirical failure modes that should be reported rather than hidden.

## 7. Conclusion

LoRGCD proposes that category discovery should use not one global geometry but a field of local metrics and tangent orientations. The resulting tangent-bundle graph distinguishes nuisance motion from locally discriminative directions and can reject near points whose tangent structures disagree. RAD distills this geometry into an ordinary embedding, GMD uses it for stable pseudo-target diffusion, and the same graph supports unknown-\(K\) estimation. Whether these theoretical advantages translate into CVPR-level empirical gains is explicitly left to the planned controlled experiments; the repository never fabricates those results.
