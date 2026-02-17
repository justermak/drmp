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

| Metric | MPD | gpmp2-rrt-prior | rrt-connect | gpmp2-uninformative |
| --- | --- | --- | --- | --- |
| n_tasks | 100 | 76 | 100 | 76 |
| n_samples | 100 | 100 | 100 | 100 |
| Time to generate n_samples | 0.506 ± 0.003 sec | 14.245 ± 2.048 sec | 6.736 ± 0.385 sec | 3.375 ± 0.064 sec |
| Success rate | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 57.24 ± 11.18% |
| Free fraction | 97.11 ± 0.88% | 97.43 ± 0.62% | 94.46 ± 2.17% | 47.14 ± 10.55% |
| Collision intensity | 0.03 ± 0.01% | 0.01 ± 0.00% | 0.04 ± 0.02% | 0.99 ± 0.25 % |
| Best path length | 2.1273 ± 0.0655 | 1.9895 ± 0.0590 | 2.4455 ± 0.0670 | 1.8593 ± 0.0627 |
| Sharpness | 0.4040 ± 0.0200 | 0.3949 ± 0.0188 | 4.4831 ± 0.1128 | 0.3226 ± 0.0318 |
| Path length | 2.4950 ± 0.0595 | 2.4334 ± 0.0740 | 3.4323 ± 0.0814 | 1.8627 ± 0.0644 |
| Waypoints variance | 0.1859 ± 0.0045 | 0.2001 ± 0.0088 | 0.2418 ± 0.0060 | 0 ± 0 |

### Extra objects

| Metric | MPD | gpmp2-rrt-prior | rrt-connect-splines | gpmp2-uninformative |
| --- | --- | --- | --- | --- |
| n_tasks | 1000 | 76 | 300 | 300 |
| n_samples | 100 | 100 | 1 | 1 |
| Time to generate n_samples | 0.586 ± 0.005 sec | 15.430 ± 2.211 sec | 2.913 ± 0.244 sec | 1.755 ± 0.016 sec |
| Success rate | 84.55 ± 2.25% | 100.00 ± 0.00% | 67.50 ± 5.17% | 38.33 ± 5.33% |
| Free fraction | 47.53 ± 1.90% | 96.54 ± 0.74% | 67.67 ± 5.33% | 38.17 ± 5.50% |
| Collision intensity | 0.66 ± 0.03% | 0.02 ± 0.01% | 2.15 ± 0.68% | 1.49 ± 0.20% |
| Best path length | 2.1757 ± 0.0233 | 1.9895 ± 0.0593 | 2.8748 ± 0.0843 | 1.8256 ± 0.0340 |
| Sharpness | 0.7123 ± 0.0151 | 0.3937 ± 0.0186 | 17074.7402 ± 3398.0365 | 386.6198 ± 68.1822 |
| Path length | 2.5409 ± 0.0226 | 2.4286 ± 0.0732 | 3.4323 ± 0.0814 | 1.8251 ± 0.0336 |
| Waypoints variance | 0.1742 ± 0.0038 | 0.1984 ± 0.0090 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

RRT_connect cp
| n_tasks | 300 |
| n_trajectories_per_task | 100 |
| Time to generate n_trajectories_per_task | 38.113 ± 2.669 sec |
| Success rate | 88.83 ± 3.50% |
| Free fraction | 64.58 ± 3.84% |
| Collision intensity | 3.62 ± 0.79% |
| Best path length | 2.2372 ± 0.0465 |
| Sharpness | 14132.7982 ± 1174.7089 |
| Path length | 2.8576 ± 0.0514 |
| Waypoints variance | 0.1991 ± 0.0070 |


exact
| n_tasks | 300 |
| n_trajectories_per_task | 100 |
| Time to generate n_trajectories_per_task | 37.157 ± 3.208 sec |
| Success rate | 64.17 ± 5.50% |
| Free fraction | 2.62 ± 0.43% |
| Collision intensity | 8.00 ± 0.80% |
| Best path length | 3.1429 ± 0.1014 |
| Sharpness | 74302.9881 ± 11933.7084 |
| Path length | 3.4818 ± 0.0937 |
| Waypoints variance | 0.1057 ± 0.0142 |

RRT-Connect + GPMP2 cp
| n_tasks | 300 |
| n_trajectories_per_task | 100 |
| Time to generate n_trajectories_per_task | 41.748 ± 2.593 sec |
| Success rate | 100.00 ± 0.00% |
| Free fraction | 87.94 ± 2.36% |
| Collision intensity | 0.21 ± 0.06% |
| Best path length | 1.9473 ± 0.0286 |
| Sharpness | 201.9304 ± 21.7101 |
| Path length | 2.3085 ± 0.0359 |
| Waypoints variance | 0.1910 ± 0.0041 |

exact
| n_tasks | 300 |
| n_trajectories_per_task | 100 |
| Time to generate n_trajectories_per_task | 40.858 ± 3.201 sec |
| Success rate | 100.00 ± 0.00% |
| Free fraction | 84.99 ± 2.57% |
| Collision intensity | 0.26 ± 0.06% |
| Best path length | 1.9658 ± 0.0288 |
| Sharpness | 306.3921 ± 24.4043 |
| Path length | 2.3748 ± 0.0354 |
| Waypoints variance | 0.2028 ± 0.0045 |