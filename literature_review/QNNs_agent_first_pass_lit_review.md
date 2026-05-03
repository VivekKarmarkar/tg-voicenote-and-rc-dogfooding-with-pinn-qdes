# QNNs & PINNs + Quaternions — Agent First Pass Literature Review

## 1. Quaternion Neural Networks (QNNs)

Well-established subfield. Multiple surveys exist:
- Zhu et al. (2019, Artificial Intelligence Review)
- 2024 comprehensive analysis in Archives of Computational Methods in Engineering
- 2024 review in Advances in Applied Clifford Algebras

Core idea: replace real-valued weights with quaternion algebra (Hamilton product) to capture inter-channel relationships with fewer parameters. Applications in speech (Parcollet et al.), image processing, and signal processing.

## 2. Quaternionic Latent Space

No papers use the exact phrase "quaternionic latent space." QNNs implicitly operate over quaternion-valued hidden representations, but nobody frames it as a latent-space contribution. This is a gap.

## 3. Unit Quaternion Outputs for Rotation

Active area. Key papers:
- Zhou et al. (2019, CVPR) "On the Continuity of Rotation Representations in Neural Networks" — argues quaternions are discontinuous, favors 6D representation
- Pavllo et al. (2018) "QuaterNet" for human motion prediction
- Zhang et al. (2020, ICML) "Quaternion Product Units"

Note: the field is moving away from quaternion outputs toward continuous representations, but quaternion outputs remain common in robotics/pose estimation.

## 4. PINNs + Quaternions

Essentially one paper:
- **TE-PINN** (arXiv:2409.16214, 2024) — "Quaternion-Based Orientation Estimation using Transformer-Enhanced Physics-Informed Neural Networks." Uses PINNs to enforce quaternion kinematics for inertial navigation. This is sensor fusion, not solving QDEs.

**No paper uses PINNs to solve quaternionic differential equations.** This is the gap our project fills.

## 5. PINNs + Complex-Valued

Exists as the closest analog:
- **CompleX-PINN** (arXiv:2502.04917, 2025) — uses complex-valued activation functions inspired by Cauchy integral theorem
- PINNs applied to complex-valued neutrino oscillation ODEs (arXiv:2604.22862, 2025)

Confirms the progression: real-valued PINNs (mature) → complex-valued PINNs (emerging) → quaternion-valued PINNs (our contribution, nonexistent).

## 6. Quaternionic DEs + Neural Networks

- Kou et al. (2018, Studies in Applied Mathematics) — established the theory of linear QDEs
- Neural network models exist for quaternion SVD
- **No one has used neural networks (PINN or otherwise) to solve QDEs directly**
- Noncommutativity of quaternion algebra is noted as the key theoretical challenge

## Bottom Line

The niche "PINNs for quaternionic differential equations" is unoccupied. The closest work is TE-PINN (sensor fusion, not DE solving) and complex-valued PINNs (lower-dimensional algebra). Our project sits at an unoccupied intersection.
