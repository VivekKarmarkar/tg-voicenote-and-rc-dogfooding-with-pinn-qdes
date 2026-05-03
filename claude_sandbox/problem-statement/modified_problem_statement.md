# Modified Problem Statement

See `problem-_tatement.md` for the full problem scope (3 IMU datasets, all non-stationary segments, inclination and heading estimates via PINN).

## Focused Sub-Problem

We solve a simpler, more tractable problem first: **Segment 23 only**.

Segment 23 is extracted from the reference dataset using the preprocessing code in `01_preprocess.py`. It spans approximately 226.3s–230.0s (3.6 seconds, 720 samples at 200 Hz).

## Goal

The PINN must estimate angular velocity components within ±0.05 rad/s envelopes for ω_x, ω_y, ω_z) at every time point

## Deliverable

### ONLY Plot: Angular Velocity Overlay

- Three gyroscope components (ω_x, ω_y, ω_z) from the provided measurement data
- Three PINN-predicted angular velocities, reconstructed via ω = 2q*⊗q̇
- Translucent ±0.05 rad/s envelope around (ω_x, ω_y, ω_z)
- PINN-predicted bold dashed lines in a distinct color
- Overlaid on the same axes

- **Pass criteria**: the PINN dashed lines must stay inside the ±0.05 rad/s envelopes for ω_x, ω_y, ω_z) at every time point

