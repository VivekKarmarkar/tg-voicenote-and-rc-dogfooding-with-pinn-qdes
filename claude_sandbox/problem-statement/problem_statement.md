Given to you are details of 3 IMU datasets (each corresponding to IMU sensor data for 1 rigid body)

Task: Produce inclination and heading angle estimates for dataset segments FROM ANGULAR VELOCITY DATA ONLY where the RIGID MODY IS NOT STATIONARY given the provided dataset by USING a PINN ONLY where PINN means PHYSICS-INFORMED-NEURAL NETWORK

Note: PRODUCE RESULTS ONLY FOR DATASET SEGMENTS WHERE THE RIGID BODY IS NOT STATIONARY

Note: Work only INSIDE the "claude_sandbox" folder

The 4 files correspond to individual IMU sensors and contain:
1. Packet counter [-]
2. Time [1e-4s]
3. Accelerometer data [m/s^2]
4. Gyroscope data
5. Orientation increment [-] (unit quaternion)
6. Velocity increment [m/s]
