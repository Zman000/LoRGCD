# Theory: LoRGCD as Tangent-Bundle Geometry for Generalized Category Discovery

## 1. Problem setup

Let \(D_l=\{(x_i,y_i)\}\) contain labeled samples from known classes \(\mathcal C_{old}\). Let \(D_u=\{x_i\}\) contain an unlabeled mixture from \(\mathcal C_{old}\cup\mathcal C_{new}\). A backbone and projection head give normalized features \(z_i\in\mathbb S^{d-1}\). GCD must assign semantic categories to every unlabeled example without labels for \(\mathcal C_{new}\).

The geometric hypothesis of LoRGCD is specific: **one homogeneous metric is generally insufficient when class manifolds have location-dependent anisotropy and tangent orientation**. Cosine distance weights every ambient direction equally; one constant-curvature model changes global geometry but remains homogeneous; global whitening/isotropy regulates one batch-level spectrum; fixed prototype geometry constrains class centers rather than sample-wise tangent structure.

LoRGCD therefore estimates a local metric *and* a local tangent orientation at each sample. Together they define a discrete approximation to a tangent-bundle affinity.

## 2. Local generative model

Around a point of class \(c\), approximate a small feature neighborhood by

\[
z=\mu_c+U_c a+\epsilon,
\qquad a\sim\mathcal N(0,\Lambda_c),
\qquad \epsilon\sim\mathcal N(0,\sigma_c^2I).
\]

\(U_c\) spans locally high-variance tangent/nuisance directions such as pose, viewpoint, illumination, articulation, or intra-class texture. Small motion normal to this local tangent can be more semantically meaningful than much larger tangent motion.

The local covariance is

\[
\Sigma_c=U_c\Lambda_cU_c^\top+\sigma_c^2I,
\]

and its inverse is the ideal standardized local metric.


## 3. Proposition 1 — one global metric cannot whiten heterogeneous local geometries

Suppose two neighborhoods have SPD covariances \(\Sigma_a\) and \(\Sigma_b\). Assume there exists one global SPD metric \(M\) that makes both neighborhoods isotropic up to scalar factors:

\[
M^{1/2}\Sigma_aM^{1/2}=c_aI,
\qquad
M^{1/2}\Sigma_bM^{1/2}=c_bI,
\]

for positive \(c_a,c_b\). Multiplying on the left and right by \(M^{-1/2}\) gives

\[
\Sigma_a=c_aM^{-1},
\qquad
\Sigma_b=c_bM^{-1},
\]

hence

\[
\Sigma_a=\frac{c_a}{c_b}\Sigma_b.
\]

Therefore a single global metric can isotropize both neighborhoods only if their covariance tensors are proportional. If local classes/poses have different eigenvectors or different relative eigenvalue spectra, no one global Mahalanobis/whitening operator can standardize all of them simultaneously.

### Consequence

This is the formal reason LoRGCD uses a metric field \(M(z)\) instead of one learned/global metric. Global isotropy and global curvature can still be useful regularizers, but they cannot, in general, remove heterogeneous local nuisance anisotropy everywhere at once.

## 4. Proposition 2 — inverse covariance amplifies low-variance class separation

Assume two nearby classes share covariance \(\Sigma\), with means \(\mu_1,\mu_2\), and let \(\Delta=\mu_1-\mu_2\). For \(M=\Sigma^{-1}\):

\[
\mathbb E[(z-z')^\top M(z-z')\mid y=y']=2d,
\]

while

\[
\mathbb E[(z-z')^\top M(z-z')\mid y\ne y']
=2d+\Delta^\top\Sigma^{-1}\Delta.
\]

### Proof

For a random vector \(v\) with mean \(m\) and covariance \(C\),

\[
\mathbb E[v^\top Mv]=\mathrm{tr}(MC)+m^\top Mm.
\]

A same-class difference has mean zero and covariance \(2\Sigma\), hence \(2\operatorname{tr}(I)=2d\). A cross-class difference has covariance \(2\Sigma\) and mean \(\Delta\), giving the additional standardized margin \(\Delta^\top\Sigma^{-1}\Delta\).

### Consequence

If a discriminative displacement lies in a locally low-variance normal direction of variance \(\sigma_\perp^2\), while nuisance tangent variance is \(\sigma_t^2\gg\sigma_\perp^2\), standardized distance magnifies that discriminative direction by approximately \(1/\sigma_\perp^2\). This yields the testable prediction that local geometry should help most as local anisotropy increases.

## 5. Sample-wise tangent and metric estimator

For each normalized feature \(z_i\), take \(k\) cosine-nearest points only as an **initial local estimation neighborhood**. Form centered offsets

\[
X_i=[z_j-z_i]_{j\in\mathcal N_i},
\qquad X_i=P_iS_iV_i^\top.
\]

The first \(r\) right singular vectors form \(U_i\in\mathbb R^{r\times d}\), an orthonormal estimate of the local tangent subspace. Local tangent variances are

\[
\lambda_{i\ell}=\frac{s_{i\ell}^2}{k-1}+\varepsilon.
\]

Set a normal variance floor

\[
\sigma_{\perp,i}^2=\rho\,\frac1r\sum_{\ell=1}^r\lambda_{i\ell},
\quad 0<\rho\le 1.
\]

The local SPD metric is

\[
M_i=U_i^\top\operatorname{diag}(\lambda_i^{-1})U_i
+\sigma_{\perp,i}^{-2}(I-U_i^\top U_i).
\]

The implementation never constructs the full \(d\times d\) matrix. For \(\delta_{ij}=z_j-z_i\):

\[
d_i^2(i,j)=
\sum_{\ell=1}^{r}\frac{\langle \delta_{ij},u_{i\ell}\rangle^2}{\lambda_{i\ell}}
+\frac{\|\delta_{ij}^{\perp}\|^2}{\sigma_{\perp,i}^2},
\]

and the symmetric standardized displacement is

\[
d_M^2(i,j)=\tfrac12(d_i^2(i,j)+d_j^2(i,j)).
\]

## 6. Proposition 3 — tangent disagreement separates intersecting manifolds

For two \(r\)-dimensional tangent subspaces with orthonormal row bases \(U_i,U_j\), define

\[
d_T^2(i,j)=r-\|U_iU_j^\top\|_F^2.
\]

If \(\theta_1,\dots,\theta_r\) are their principal angles, then

\[
d_T^2(i,j)=\sum_{\ell=1}^{r}\sin^2\theta_\ell.
\]

Therefore \(d_T^2=0\) iff the subspaces coincide, while it is positive when the local tangents differ.

### Proof

The singular values of \(U_iU_j^\top\) are \(\cos\theta_\ell\). Thus

\[
\|U_iU_j^\top\|_F^2=\sum_\ell \cos^2\theta_\ell,
\]

which directly gives the result.

### Why this matters for discovery

At a manifold intersection or near-touching region, two points from different semantic manifolds can have arbitrarily small Euclidean/cosine separation. A purely positional neighborhood criterion can therefore connect them. If the manifolds have different tangents, \(d_T\) stays positive even as positional distance tends to zero. The tangent term supplies information unavailable to any point-distance-only metric.

## 7. Tangent-Bundle Riemannian Affinity (TBRA)

The two energies have different units/scales, so LoRGCD normalizes each by a robust local median rather than introducing another tuned mixing coefficient:

\[
E_{ij}=\frac{d_M^2(i,j)}{\operatorname{med}(d_M^2)}
+\frac{d_T^2(i,j)}{\operatorname{med}(d_T^2)}.
\]

The initial cosine neighbors are **not** forced to remain graph neighbors. After estimating \(U_i\) and \(M_i\), graph neighbors are re-selected using \(E_{ij}\). The affinity is

\[
A_{ij}\propto \exp(-E_{ij}/\tau_g).
\]

For mini-batches, all pairwise candidates are affordable. For full-dataset spectral estimation, the code uses a larger cosine candidate pool (default \(4k\)) and re-ranks that pool by the same tangent-bundle energy, keeping the graph sparse and scalable.

Known-class labeled points can optionally inject must-links for identical known labels and cannot-links for different known labels. No novel labels are used.

## 8. Proposition 4 — consistency of the local tangent estimate

Assume data are sampled from a compact \(C^2\) manifold with bounded curvature and isotropic observation noise. As sample size grows, let the local neighborhood radius \(h_n\to0\), \(k\to\infty\), and \(k/n\to0\). Under a nonzero tangent/normal eigengap, local PCA converges to the tangent space.

This is a standard local-PCA/manifold-estimation result rather than a new theorem. LoRGCD's contribution is how that estimate is converted into a detached geometric teacher for GCD and combined with tangent-bundle consistency.

## 9. RAD — Riemannian Affinity Distillation

A local graph is useful during training, but a CV system should retain simple inference. Let the student cosine distribution be

\[
S_{ij}=\operatorname{softmax}_j(z_i^\top z_j/\tau_s).
\]

LoRGCD distills the tangent-bundle graph into the embedding:

\[
\mathcal L_{RAD}=-\sum_i\sum_j A_{ij}\log S_{ij}
=\mathrm{KL}(A\|S)+\text{const.}
\]

Thus the expensive local geometry is a training-time teacher; inference remains the original backbone/projector/classifier.

## 10. GMD — Geometry-aware Mean-Shift Distillation

The second use of \(A\) is a graph-diffusion approximation to local manifold smoothing:

\[
\tilde z_i=\operatorname{norm}\!\left(\sum_jA_{ij}z_j\right).
\]

Teacher pseudo-targets are generated from \(\tilde z_i\), and the opposite augmentation predicts them. We deliberately call this **geometry-aware mean shift**, not an exact geodesic solver: the operation is one or a few steps of anisotropic graph diffusion induced by the estimated local geometry.

This distinction is important theoretically and for reviewer clarity.


## 11. Proposition 5 — local anisotropic mean shift is a natural-gradient density step

Consider the positional kernel only and freeze the local SPD metric \(M_i\) while updating sample \(z_i\). Define the local KDE

\[
\hat p_i(z)=\sum_j \exp\!\left[-\frac{1}{2h^2}(z-z_j)^\top M_i(z-z_j)\right].
\]

Let normalized kernel weights be \(a_{ij}(z)\). Differentiating the log density yields

\[
\nabla_z\log \hat p_i(z)
=\frac{1}{h^2}M_i\left(\sum_j a_{ij}(z)z_j-z\right).
\]

Therefore the mean-shift vector satisfies

\[
\underbrace{\sum_j a_{ij}(z)z_j-z}_{m_i(z)-z}
=h^2M_i^{-1}\nabla_z\log\hat p_i(z).
\]

Because the Riemannian/natural gradient under metric \(M_i\) is \(M_i^{-1}\nabla f\), anisotropic mean shift performs a natural-gradient ascent step on local feature density. LoRGCD's GMD uses the TBRA graph, so the exact identity applies to its positional local-metric component; tangent disagreement and graph symmetrization act as additional topology-aware edge corrections.

### Consequence

GMD is not merely averaging neighboring predictions. Under the local metric model it moves features toward density modes while respecting anisotropic nuisance scales, which is exactly the behavior desired for category discovery.

## 12. Full objective

A practical SimGCD-style objective is

\[
\mathcal L=
\lambda_{inst}\mathcal L_{NTXent}
+\lambda_{supcon}\mathcal L_{SupCon}
+\mathcal L_{self\text{-}distill}
+\lambda_{sup}\mathcal L_{CE}
-\lambda_H H(\bar p)
+\lambda_{RAD}\mathcal L_{RAD}
+\lambda_{GMD}\mathcal L_{GMD}.
\]

Only the last two losses and their common tangent-bundle teacher are new. The geometry is computed under `torch.no_grad()` so the network cannot reduce the loss by directly manipulating singular vectors/metric parameters.

## 13. Proposition 6 — the same graph supports unknown-K estimation

For an ideal affinity graph with exactly \(K\) disconnected class components, the normalized Laplacian

\[
L=I-D^{-1/2}AD^{-1/2}
\]

has exactly \(K\) zero eigenvalues. Hence the eigengap \(\lambda_{K+1}-\lambda_K\) is positive when each component is connected.

Under a symmetric perturbation \(E\), Weyl's inequality bounds eigenvalue movement by \(\|E\|_2\). Therefore a sufficiently large ideal eigengap remains identifiable under small cross-class graph contamination. This motivates using the same LoRGCD graph for category-count estimation. The empirical claim is falsifiable: TBRA must reduce cross-class graph mass and produce a more stable eigengap than cosine/global-Mahalanobis graphs.

## 14. What is actually novel vs classical

Classical ingredients: local PCA, inverse covariance/Mahalanobis standardization, Grassmann principal-angle distance, graph diffusion, spectral eigengaps.

The proposed contribution is their **GCD-specific coupling**:

1. sample-wise inverse-covariance geometry from unlabeled neighborhoods;
2. tangent-space agreement as an independent edge criterion;
3. geometric re-ranking rather than only reweighting a fixed cosine graph;
4. distillation of that graph into a standard cosine representation (RAD);
5. the same graph for pseudo-target smoothing (GMD) and unknown-K estimation;
6. no added inference parameters.

A literature search through 1 Oct 2026 did not surface this exact combination. That is evidence for a promising gap, not proof of novelty; a final submission still needs equation-level prior-art checking.

## 15. Falsifiable predictions

A theory-backed paper should try to disprove itself. The method predicts:

- gains increase with local covariance anisotropy;
- gains increase with tangent disagreement near incorrect cosine neighbors;
- fine-grained datasets benefit more than coarse datasets;
- removing the Grassmann tangent term hurts most near manifold intersections/close classes;
- \(\rho\to1\) weakens the normal-direction advantage;
- too-small \(k\) makes SVD noisy, while too-large \(k\) violates locality;
- local TBRA should beat one global Mahalanobis metric if heterogeneity is essential;
- TBRA should improve graph purity and unknown-\(K\) eigengap stability, not only final accuracy;
- when local neighborhoods are already isotropic and pure, LoRGCD should offer little gain.

These predictions define the main ablations and diagnostic figures in `EXPERIMENTS.md`.
