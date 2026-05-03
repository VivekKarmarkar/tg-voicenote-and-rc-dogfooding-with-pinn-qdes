# Additional Context

## 1. About PINNs: Karniadkis wrote a famous Nature Reviews PINN paper which augments his seminal PINN paper with Raissi and the point to note is there is no rigid definition of PINN other than incorporating Physics into NNs. Our approach aligns with the Raissi paper and is in spirit of the Nature Review paper - in other words, don't stick to some method from a paper BLINDLY - UNDERSTAND YOUR PROBLEM PROPERLY

## 2. For Segment 23:

- The **first time point** of the extracted phase has a corresponding reference quaternion → use as the **initial condition**
- The **last time point** of the extracted phase has a corresponding reference quaternion → use as a **boundary condition**
- Together, these two reference quaternions at the endpoints form the BCs

## 3. The Quaternionic Differential Equation — Use Gyroscope Data Only

The quaternionic kinematic ODE is: **q̇ = ½ q ⊗ ω**

But the equation must be rearranged so that **ω is isolated on the left-hand side**:

**ω = 2 q* ⊗ q̇**

where q* is the quaternion conjugate.

- **Only gyroscope data is used**
