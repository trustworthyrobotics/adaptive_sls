Vendored from `google-deepmind/mujoco_menagerie`, directory
`bitcraze_crazyflie_2`, at commit:

    8161bba264d7fa7c99ca301e91e7fb44737676ad

The upstream XML describes the Crazyflie 2 family at 27 g. The adaptive MPC
experiment targets the non-brushless Crazyflie 2.1/2.1+ and scales the model's
mass and diagonal inertia at runtime to the 29 g nominal value from Bitcraze's
Crazyflie 2.1 datasheet before applying `--true-mass-scale`.
