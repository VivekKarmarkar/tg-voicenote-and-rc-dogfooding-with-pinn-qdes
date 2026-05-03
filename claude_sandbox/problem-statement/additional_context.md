# Additional Context

## 1. Data Preprocessing — Segment Stationary vs Non-Stationary Phases

The rigid body has 3 IMU sensors attached at various orientations. The time series consists of alternating phases: stationary → movement → stationary → movement → stationary, etc.

- **Stationary** = zero angular velocity (from gyroscope data)
- Extract the **non-stationary segments only**, with a small padding before and after each segment so the transition from stationary to movement (and back) is visible
- Only produce PINN results for these non-stationary segments

## 2. The Quaternionic Differential Equation — Use Gyroscope Data Only

The quaternionic kinematic ODE is: **q̇ = ½ q ⊗ ω**

But the equation must be rearranged so that **ω is isolated on the left-hand side**:

**ω = 2 q* ⊗ q̇**

where q* is the quaternion conjugate.

This means:
- The PINN's neural network learns the quaternion trajectory q(t)
- The derivative q̇(t) is obtained via automatic differentiation
- The physics loss compares the reconstructed ω (from q and q̇) against the measured gyroscope data
- **Only gyroscope data is used** — no accelerometer, no velocity increments, no orientation increments