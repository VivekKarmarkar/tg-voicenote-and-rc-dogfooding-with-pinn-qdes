# Modified Problem Statement 2

See `problem_statement.md` for the full problem scope (3 IMU datasets, all non-stationary segments, inclination and heading estimates via PINN).

## Focused Sub-Problem

- Solve the problem for the datasets in the "test_files" subfolder
- Focus on the datasets in the "test_files" subfolder
- The datasets currently included are seg23, trial_{idx} with idx = [000, 006, 009, 011, 016]

## Goal

The PINN must estimate angular velocity components within ±0.1 rad/s envelopes for ω_x, ω_y, ω_z) at every time point and the PINN MUST BE DESIGNED SUCH THAT THE ALGORITHM FINISHES OPTIMIZATION IN LESS THAN 5 MINUTES.

## Rules
- You CAN ONLY TOUCH 05_solve_segment_general_pinn.py
- You ARE SUPPOSED TO MAKE ONE ALGORITHM THAT WORKS FOR ALL CASES - NOT CREATE A MONOLITHIC DUMP WITH THOUSANDS OF CASE-SPECIFIC SETTINGS - MAKE ONE ALGORITHM
- YOU ARE REQUIRED TO USE JAX
- YOU ARE REQUIRED TO NOT JUST USE JAX BUT YOU ARE ALSO REQUIRED TO USE AUTOMATIC DIFFERENTATION IN JAX
- DO NOT MAKE RADICAL CHANGES THAT COULD DEFY AND BREAK EVERYTHING THAT WORKS UP TILL THAT POINT
- MAKE CHANGES THAT ABSOFUCKINGLUTELY DO NOT BREAK A SINGLE FUCKING THING WHICH WORK BUT INSTEAD VERY CAREFULLY AND EXPRESSLY ARE DESIGNED TO CREATE HIGHLY FLEXIBLE ALGORITHMS THAT HANDLE FAILURE CASES WHILE MAINTAINING OR INCREASING ACCURACY ON SUCCESSFUL CASES

## HARD CONSTRAINT

#### You need to solve this problem in 40 minutes total

- You need to create a log called "attempt_history.md"
- This log will have 2 column headers {attempt_index, attempt_time (mins:seconds), cumulative_attempt_time (mins:seconds)}
- You will initialize cumulative_attempt_time to 0 seconds before the GAME starts
- Then you will add attempt_time for a given attempt to cumulative_attempt_time
- You are allowed to play until the cumulative_attempt_time hits 40 minutes
- Cumulative_attempt_time = 40 minutes is the **HARD CONSTRAINT**

## Test

We are testing to see if CLAUDE CAN SOLVE THIS AUTONOMOUSLY - VIVEK ALREADY HAS HIS VERSION

## Deliverable

### ONLY Plot: Angular Velocity Overlay (comprehensive)

- Three gyroscope components (ω_x, ω_y, ω_z) from the provided measurement data
- Three PINN-predicted angular velocities, reconstructed via ω = 2q*⊗q̇
- Translucent ±0.1 rad/s envelope around (ω_x, ω_y, ω_z)
- PINN-predicted bold dashed lines in a distinct color
- Overlaid on the same axes

- **Pass criteria**: the PINN dashed lines must stay inside the ±0.1 rad/s envelopes for ω_x, ω_y, ω_z) at every time point

