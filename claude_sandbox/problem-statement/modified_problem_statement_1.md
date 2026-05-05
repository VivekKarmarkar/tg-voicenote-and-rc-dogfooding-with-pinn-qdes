# Modified Problem Statement

See `problem_statement.md` for the full problem scope (3 IMU datasets, all non-stationary segments, inclination and heading estimates via PINN).

## Focused Sub-Problem

We solve a simpler, more tractable problem first: **Segment 23 and user-provided test segments only**.

## Goal

The PINN must estimate angular velocity components within ±0.05 rad/s envelopes for ω_x, ω_y, ω_z) at every time point and the PINN MUST BE DESIGNED SUCH THAT THE ALGORITHM FINISHES OPTIMIZATION IN LESS THAN 3 MINUTES.

## Rule
You CAN ONLY TOUCH 05_solve_segment_general_pinn.py

## HARD CONSTRAINT

#### You need to solve this problem in 15 minutes total

- You need to create a log called "attempt_history.md"
- This log will have 2 column headers {attempt_index, attempt_time (mins:seconds), cumulative_attempt_time (mins:seconds)}
- You will initialize cumulative_attempt_time to 0 seconds before the GAME starts
- Then you will add attempt_time for a given attempt to cumulative_attempt_time
- You are allowed to play until the cumulative_attempt_time hits 15 minutes
- Cumulative_attempt_time = 30 minutes is the **HARD CONSTRAINT**

## Test

We are testing to see if CLAUDE CAN SOLVE THIS AUTONOMOUSLY - VIVEK ALREADY HAS HIS VERSION

## Deliverable

### ONLY Plot: Angular Velocity Overlay

- Three gyroscope components (ω_x, ω_y, ω_z) from the provided measurement data
- Three PINN-predicted angular velocities, reconstructed via ω = 2q*⊗q̇
- Translucent ±0.05 rad/s envelope around (ω_x, ω_y, ω_z)
- PINN-predicted bold dashed lines in a distinct color
- Overlaid on the same axes

- **Pass criteria**: the PINN dashed lines must stay inside the ±0.05 rad/s envelopes for ω_x, ω_y, ω_z) at every time point

