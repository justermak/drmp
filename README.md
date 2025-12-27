# Project on exproling applications of diffusion models to robot motion planning

---
### Current results

Reimplemented MPD https://arxiv.org/abs/2308.01557, optimized infrastructure

---
## Baseline models evaluation on random data

### Original MPD:

-------- TEST SPLIT --------
| Name | Result |
| --- | --- |
| n_tasks | 100 |
| n_samples | 100 |
| Time to generate n_samples | 0.491 ± 0.022 sec |
| Success rate | 82.00 ± 76.84% |
| Free fraction | 45.48 ± 56.00% |
| Collision intensity | 0.65 ± 0.85 % |
| Best path length | 2.1641 ± 0.2658 |
| Sharpness | 2.5152 ± 0.4418 |
| Path length | 2.5103 ± 0.2875 |
| Waypoints variance | 0.1738 ± 0.0373 |
