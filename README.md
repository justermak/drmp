# Project on exproling applications of diffusion models to robot motion planning

---
### Current results

Reimplemented MPD https://arxiv.org/abs/2308.01557, optimized infrastructure

---
## Baseline models evaluation on random data

### No extra objects

| Metric | diffusion-original | gpmp2-rrt-prior | rrt-connect | gpmp2-uninformative |
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
| Waypoints variance | 0.1859 ± 0.0045 | 0.2001 ± 0.0088 | 0.2418 ± 0.0060 | nan ± nan |

### Extra objects

| Metric | diffusion-original | gpmp2-rrt-prior | rrt-connect | gpmp2-uninformative |
| --- | --- | --- | --- | --- |
| n_tasks | 100 | 76 | 100 | 76 |
| n_samples | 100 | 100 | 100 | 100 |
| Time to generate n_samples | 0.506 ± 0.003 sec | 15.430 ± 2.211 sec | 6.736 ± 0.385 sec | 2.967 ± 0.026 sec |
| Success rate | 82.50 ± 7.50% | 100.00 ± 0.00% | 100.00 ± 0.00% | 42.76 ± 11.18% |
| Free fraction | 50.73 ± 6.75% | 96.54 ± 0.74% | 94.40 ± 2.15% | 30.55 ± 9.63% |
| Collision intensity | 0.64 ± 0.11% | 0.02 ± 0.01% | 0.04 ± 0.02% | 1.33 ± 0.26% |
| Best path length | 2.0954 ± 0.0593 | 1.9895 ± 0.0593 | 2.4455 ± 0.0670 | 1.8724 ± 0.0745 |
| Sharpness | 0.6511 ± 0.0332 | 0.3937 ± 0.0186 | 4.4831 ± 0.1128 | 0.2900 ± 0.0328 |
| Path length | 2.4474 ± 0.0654 | 2.4286 ± 0.0732 | 3.4323 ± 0.0814 | 1.8736 ± 0.0744 |
| Waypoints variance | 0.1841 ± 0.0097 | 0.1984 ± 0.0090 | 0.2418 ± 0.0060 | 0 ± 0 |
