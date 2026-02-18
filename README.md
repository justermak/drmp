# DRMP (Diffusion robot motion planning)
Project on exproling applications of generative models (especially diffusion) to robot motion planning

### Installation

```
uv venv --python 3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .

```
---
## Experiments

Setup: All classical algorithms are just ran on both versions of the environment. Learning base algorithms are trained only on the base environment and tested on both versions. Since they don't take the environment as an input, runing them on the environment with extra obstacles tests their ability to generate diverse tragectories that can be optimized 

### EnvDense2D

Map size: $[-1; 1]^2$

Robot: Sphere2D (circle of radius 0.1)

Objects: 30 circles + smoothed squares

Extra objects: 7 circles + smoothed squares

### Base

| Metric | DRMP (ours) | MPD | 
| --- | --- | --- | 
| n_tasks | 300 | 300 | 
| n_samples | 100 | 100 | 
| Time to generate n_samples | 0.066 ± 0.001 sec | 0.438 ± 0.005 sec | 
| Success rate | 100.00 ± 0.00% | 100.00 ± 0.00% | 
| Free fraction | 42.69 ± 1.45% | 96.15 ± 1.10% |
| Collision intensity | 3.75 ± 0.14% | 0.05 ± 0.02% | 
| Best path length | 2.0491 ± 0.0328 | 2.0949 ± 0.0325 | 
| Sharpness | 288.2304 ± 9.6719 | 3409.4662 ± 566.0481 |
| Path length | 2.3721 ± 0.0342 | 2.4811 ± 0.0343 | 
| Waypoints variance | 0.1730 ± 0.0033 | 0.1871 ± 0.0030 |

### Extra objects

| Metric | DRMP(ours) | MPD | gpmp2-rrt-prior | rrt-connect-splines |  rrt-connect-splines | gpmp2-uninformative |
| --- | --- | --- | --- | --- | --- | --- |
| n_tasks | 300 | 1000 | 300 | 300 | 300 | 300 |
| n_samples | 100 | 100 | 100 | 1 | 100 | 1 |
| Time to generate n_samples | 0.067 ± 0.001 sec | 0.586 ± 0.005 sec | 41.748 ± 2.593 sec | 2.913 ± 0.244 sec | 38.113 ± 2.669 sec | 1.755 ± 0.016 sec |
| Success rate | 98.83 ± 1.17% | 84.55 ± 2.25% | 100.00 ± 0.00% | 67.50 ± 5.17% | 88.83 ± 3.50% | 38.33 ± 5.33% |
| Free fraction | 18.41 ± 1.47% | 47.53 ± 1.90% | 87.94 ± 2.36% | 67.67 ± 5.33% | 64.58 ± 3.84% | 38.17 ± 5.50% |
| Collision intensity | 7.01 ± 0.25% | 0.66 ± 0.03% | 0.21 ± 0.06% | 2.15 ± 0.68% | 3.62 ± 0.79% | 1.49 ± 0.20% |
| Best path length | 2.0776 ± 0.0331 | 2.1757 ± 0.0233 | 1.9473 ± 0.0286 | 2.8748 ± 0.0843 | 2.2372 ± 0.0465 | 1.8256 ± 0.0340 |
| Sharpness | 329.6351 ± 11.3132 | 0.7123 ± 0.0151 | 201.9304 ± 21.7101 | 17074.7402 ± 3398.0365 | 14132.7982 ± 1174.7089 | 386.6198 ± 68.1822 |
| Path length | 2.2844 ± 0.0331 | 2.5409 ± 0.0226 | 2.3085 ± 0.0359 | 3.4323 ± 0.0814 | 2.8576 ± 0.0514 | 1.8251 ± 0.0336 |
| Waypoints variance | 0.1340 ± 0.0062 | 0.1742 ± 0.0038 | 0.1910 ± 0.0041 | 0.0000 ± 0.0000 | 0.1991 ± 0.0070 | 0.0000 ± 0.0000 |
