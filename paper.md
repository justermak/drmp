# Motion Planning Diffusion with Shortcut Models for Efficient Robot Trajectory Generation

## Abstract
Motion planning in complex environments remains a challenging problem in robotics, requiring algorithms that can generate collision-free and smooth trajectories efficiently. While recent approaches using diffusion models, such as Motion Planning Diffusion (MPD), have shown promise in learning trajectory distributions and acting as effective priors for optimization, they typically require multiple denoising steps, limiting their real-time applicability. This work introduces DRMP (Diffusion Robot Motion Planning), a novel approach that accelerates inference by orders of magnitude through specific architectural changes and a shortcut training objective. Our method, DRMP, achieves path generation times as low as 0.016s on 2D environments—a 20x-30x speedup over the baseline MPD—while maintaining comparable success rates and trajectory quality. We demonstrate the efficacy of our approach on various 2D environments. Furthermore, we extend our evaluation to robots with non-trivial shapes (L-shaped robots), demonstrating the method's capability to handle complex kinematic and geometric constraints beyond simple point-mass or circular robots.

## 1. Introduction
Robot motion planning is fundamentally the problem of finding a valid sequence of states connecting a start configuration to a goal configuration while avoiding obstacles and satisfying kinematic constraints. Traditional methods like RRT* or PRM are complete but can be computationally expensive and produce jerky paths. Optimization-based methods (e.g., GPMP2, CHOMP) produce smooth trajectories but are prone to local minima.

Recently, Generative AI has been applied to this domain. Motion Planning Diffusion (MPD) [Carvalho et al., 2023] proposes using diffusion models to learn priors over trajectories. By training on a dataset of valid demonstrations, MPD learns to generate valid paths and can be used as a prior for optimization algorithms. However, diffusion models are inherently slow due to their iterative denoising process.

Our work builds upon MPD but focuses on the critical bottleneck of inference speed. We propose a Shortcut diffusion model that learns to traverse the reverse diffusion process in fewer, larger steps. By conditioning the model on the step size $dt$ and employing a shortcut training objective, we enable high-quality one-step trajectory generation.

## 2. Methodology

### 2.1. Problem Formulation
We consider the problem of generating a collision-free trajectory $\boldsymbol{\tau} = [\mathbf{q}_0, \mathbf{q}_1, \dots, \mathbf{q}_H]$ of horizon $H$, where $\mathbf{q}_i \in \mathbb{R}^d$ is the robot configuration at step $i$. We aim to train a model to learn a distribution $p(\boldsymbol{\tau} | \mathbf{q}_{start}, \mathbf{q}_{goal})$ for a single environment.

The trained model could then be applied to generate trajectories in the following 2-step process:
1. Sample a batch of trajectories from the learned distribution
2. Run a short optimization procedure using collision avoidance and optimality promoting costs

A key idea is to collect a dataset of diverse trajectories and train a model that is capable of learning the entire distribution without losing information by collapsing to the most popular trajectories. This enables generalization to environments with additional obstacles.

### 2.2. Data Collection
The quality and diversity of the training dataset are crucial for the performance of the diffusion model. We generate a large dataset of valid trajectories using a sampling-based planner followed by optimization. For each environment, we sample random start and goal configurations. We then use **RRT-Connect**, a bidirectional sampling-based algorithm, to find an initial collision-free path. RRT-Connect is effective at exploring the configuration space but often produces jagged and suboptimal paths.

To refine these initial paths, we apply trajectory optimization. We experiment with two approaches:
1.  **Gradient Descent**: A simple optimization minimizing collision avoidance and smoothness costs.
2.  **GPMP2 (Gaussian Process Motion Planner 2)**: A probabilistic inference-based optimizer that treats the trajectory as a Gaussian Process and optimizes for smoothness and collision avoidance.

This two-stage process yields a dataset of smooth, collision-free trajectories that cover various homotopy classes, providing a rich distribution for the diffusion model to learn.

### 2.3. Trajectory Parameterization
Instead of training the diffusion model on the raw trajectory waypoints $\boldsymbol{\tau} = [\mathbf{q}_0, \ldots, \mathbf{q}_H]$, we adopt a lower-dimensional representation using B-Splines, as proposed in [1]. We parameterize the trajectory as a B-Spline curve defined by a set of control points $\mathbf{c} = [\mathbf{c}_0, \ldots, \mathbf{c}_K]$, where $K < H$.

This parameterization offers several advantages:
1.  **Efficiency**: It reduces the dimensionality of the learning problem, making training faster and requiring fewer samples.
2.  **Smoothness**: B-Splines inherently guarantee $C^2$ continuity, ensuring that the generated trajectories are smooth and dynamically feasible.
3.  **Local Control**: B-Splines provide local control over the curve, which is beneficial for learning local collision avoidance.

During training, we fit a B-Spline to each ground-truth trajectory in our dataset and train the diffusion model to essentially predict the distribution of these control points. At inference time, the model generates control points which are then evaluated to produce the dense waypoint trajectory.

### 2.4. Architecture Overview
We employ a **1D Temporal U-Net** architecture adapted for trajectory generation, similar to MPD [1], but enhanced to support variable step-size denoising through shortcut models.

- **Backbone**: The core network is a U-Net operating over the temporal dimension of the trajectory. It processes the input trajectory $\mathbf{x}_t \in \mathbb{R}^{B \times H \times D}$ (where $B$ is the batch size, $H$ is the horizon, and $D$ is the state dimension) through a hierarchical structure.
    - **Encoder variables**: The contracting path consists of a sequence of downsampling levels. Each level comprises two 1D ResNet blocks followed by a 1D convolution with stride 2 for downsampling.
    - **Bottleneck**: The lowest resolution features are processed by a bottleneck containing a ResNet block, a global 1D Self-Attention layer with multiple heads to capture long-range temporal dependencies, and another ResNet block.
    - **Decoder**: The expanding path mirrors the encoder, using 1D Transposed Convolutions for upsampling. Skip connection features from the encoder are concatenated with the upsampled features before processing by ResNet blocks.
    - **ResNet Blocks**: The fundamental building block consists of two 1D convolutions with kernel size 5, Group Normalization (groups=8), and Mish activations. Residual connections are added when input and output dimensions match.

- **Conditioning Mechanisms**: The network is conditioned on the diffusion state and the planning problem:
    - **Timestep ($t$)**: The current diffusion time $t$ is encoded using sinusoidal positional embeddings and processed by an MLP. This embedding is injected into every ResNet block using a Feature-wise Linear Modulation (FiLM) mechanism, which applies a learned scale and shift to the normalized feature maps.
    - **Step-Size ($dt$)**: To implement the shortcut model, we condition the network on the step size $dt$. We compute a separate sinusoidal embedding for $dt$, project it via an MLP, and add it to the timestep embedding vector. This allows the network to adapt its denoising prediction based on the magnitude of the shortcuts taken during the reverse process.
    - **Planning Context**: The start $\mathbf{q}_{start}$ and goal $\mathbf{q}_{goal}$ configurations are concatenated and encoded via an MLP. This context embedding is added to the time embedding, guiding the generation to satisfy the specific start and goal constraints.

### 2.5. Shortcut Training
To enable fast sampling, we train the model to approximate the result of multiple small diffusion steps with a single large step [2]. We define a composite loss function $\mathcal{L} = \mathcal{L}_{base} + \mathcal{L}_{bootstrap}$.

1.  **Base Loss ($\mathcal{L}_{base}$)**: This is the standard diffusion loss (MSE) for a single step ($dt=1$). It ensures the model remains grounded in the original diffusion process.
$$ \mathcal{L}_{base} = || \epsilon - \epsilon_\theta(\mathbf{x}_t, t, dt=1) ||^2 $$

2.  **Bootstrap Loss ($\mathcal{L}_{bootstrap}$)**: This term enforces consistency between taking one step of size $dt$ and two steps of size $dt/2$.
    1.   We sample a step size $dt = 2^k$ for $k \in \{1, \dots, \log_2 T\}$.
    2.   We generate a "target" prediction by running the model twice with step size $dt/2$ (starting from $\mathbf{x}_t$, predicting $\mathbf{x}_{t-dt/2}$, then $\mathbf{x}_{t-dt}$). This is done without gradients for the target generation.
    3.   We generate a "student" prediction by running the model once with step size $dt$.
    4.   The loss is the MSE between the student's prediction and the target.

This shortcut training allows the model to progressively learn to skip steps, eventually enabling $T$-step generation in a single forward pass ($dt=T$).

### 2.6. Optimization Costs
To further refine the quality of the generated trajectories, we employ a gradient-based optimization phase. This phase minimizes a composite cost function $J(\mathbf{\xi})$ with respect to the trajectory parameters $\mathbf{\xi}$ (either dense waypoints or B-Spline control points). The objective function is a weighted sum of four components:

1.  **Obstacle Cost**: Ensures collision-free paths by penalizing configurations that penetrate the environment's obstacles.
2.  **Velocity Cost**: Regularizes the motion by penalizing high joint velocities, ensuring the trajectory remains within dynamic limits.
3.  **Acceleration Cost**: Promotes smoothness by penalizing rapid changes in velocity.
4.  **Jerk Cost**: Encourages high-quality, comfortable motion by minimizing the rate of change of acceleration.

The optimization is performed using **gradient descent**. At each step, we calculate the gradient of the total cost $\nabla_{\mathbf{\xi}} J$ using automatic differentiation (PyTorch autograd) and update the trajectory: $\mathbf{\xi} \leftarrow \mathbf{\xi} - \nabla_{\mathbf{\xi}} J$. 

We made several observations related to the optimization procedure. Firstly, we compared two different ways to apply gradient updates:
1. Compute gradients with respect to control points.
2. Compute dense trajectories from B-Splines and then compute gradients with respect to dense points.

We found that in the second approach, it is much harder to promote trajectory smoothness using joint velocities, acceleration, and jerk costs, and it often requires more optimization steps to achieve results comparable to the first approach.

Secondly, we observed that since the model is already trained on smooth trajectories, it doesn't require many optimization steps to achieve nearly optimal levels of smoothness. In fact, in our experiments, we could reduce the number of optimization steps to just 2 without a significant drop in the quality of generated trajectories.

Thirdly, we found that given equal optimization budgets (e.g., equal number of optimization steps), the approach proposed in MPD [1]—namely, performing optimization during the last couple of steps of the diffusion denoising process—performs worse than the straightforward 'sample then optimize' approach in terms of success rate and resulting trajectory smoothness.

## 3. Evaluation

We compare our solution to multiple baselines that include sampling-based, optimization-based, learned, and combined approaches listed below. For each environment, there are two sets of obstacles: base and extra. The only difference between them is that learning-based algorithms are trained on trajectories generated in environments with only base obstacles.

Algorithms:

- gpmp2 - GPMP2. Starts from a straight-line trajectory connecting start and goal.
- grad - Gradient descent on a linear combination of cost functions. Starts from a straight-line trajectory connecting start and goal.
- grad-splines - The same algorithm but run in the space of B-Spline control points. Starts from a straight-line trajectory connecting start and goal, represented as B-Spline control points.
- rrtconnect - RRT-Connect.
- rrtconnect-spline - Uses positions generated by RRT-Connect as control points for a B-Spline.
- rrtconnect-gpmp2 - Runs RRT-Connect, then smooths the trajectory using a B-Spline, picks N points evenly on the B-Spline, and runs GPMP2 on them.
- rrtconnect-grad - Same as above, but runs Gradient Descent-based optimization.
- rrtconnect-grad-splines - After smoothing trajectories generated by RRT-Connect with a B-Spline, optimizes the resulting control points using Gradient Descent.
- mpd - Original models from MPD [1] with original optimization guides.
- mpd-reimpl - Models trained similarly to MPD but with a slightly different architecture and different data.
- mpd-splines-reimpl - Models trained similarly to MPD with B-Spline trajectory parameterization but with a slightly different architecture and different data.
- drmp (ours) - Our model.
- drmp-grad-splines - Our model + Gradient Descent-based optimization from rrtconnect-grad-splines.

Each environment is normalized to $[-1, 1]^2$.

For model training, we collected one dataset per environment consisting of 2000 random start-goal configurations and 50 trajectories generated for each, amounting to 100,000 trajectories. Trajectories in collision with obstacles were filtered out. 

We chose a horizon of $64$ for models trained on dense trajectories and $24$ for models trained on B-Spline control points, except for experiments with the L-shaped robot where we doubled both numbers.  

## RobotSphere2D

Simple circular robot with radius 0.01

### EnvSimple2D

<img src="imgs/EnvSimple2D.png" width="600">

#### Base

| algorithm               | success_rate   | time          | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:--------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 99.50 ± 0.50   | 2.770 ± 0.009 | 99.50 ± 0.01            | 99.97 ± 0.00      | 1.7722 ± 0.0240    | 1.7731 ± 0.0238   | 65.8050 ± 6.3360         | 0.0000 ± 0.0000    |
| grad                    | 73.67 ± 5.00   | 0.962 ± 0.007 | 73.33 ± 0.05            | 99.00 ± 0.00      | 1.8568 ± 0.0298    | 1.8575 ± 0.0296   | 30698.6974 ± 1336.1295   | 0.0000 ± 0.0000    |
| grad-splines            | 99.50 ± 0.50   | 1.022 ± 0.003 | 99.50 ± 0.01            | 99.99 ± 0.00      | 1.7725 ± 0.0244    | 1.7728 ± 0.0241   | 6.7795 ± 0.6213          | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 0.620 ± 0.041 | 99.70 ± 0.00            | 100.00 ± 0.00     | 1.9279 ± 0.0291    | 2.6477 ± 0.0388   | 697444.5093 ± 10704.7196 | 0.0234 ± 0.0008    |
| rrtconnect-spline       | 100.00 ± 0.00  | 1.388 ± 0.094 | 89.50 ± 0.01            | 99.66 ± 0.00      | 1.8393 ± 0.0257    | 2.3451 ± 0.0351   | 409.6015 ± 38.0385       | 0.0327 ± 0.0011    |
| rrtconnect-gpmp2        | 100.00 ± 0.00  | 3.244 ± 0.031 | 99.99 ± 0.00            | 100.00 ± 0.00     | 1.7592 ± 0.0234    | 1.9236 ± 0.0276   | 5.7036 ± 0.4621          | 0.0137 ± 0.0008    |
| rrtconnect-grad         | 100.00 ± 0.00  | 1.505 ± 0.032 | 99.97 ± 0.00            | 100.00 ± 0.00     | 1.9367 ± 0.0244    | 2.4311 ± 0.0312   | 46055.1505 ± 921.3174    | 0.0325 ± 0.0011    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 1.601 ± 0.033 | 100.00 ± 0.00           | 100.00 ± 0.00     | 1.7731 ± 0.0233    | 2.0613 ± 0.0270   | 11.3433 ± 0.4370         | 0.0274 ± 0.0010    |
| mpd                     | 100.00 ± 0.00  | 0.446 ± 0.006 | 98.74 ± 0.00            | 99.99 ± 0.00      | 1.8418 ± 0.0255    | 2.2170 ± 0.0305   | 2277.8920 ± 346.1424     | 0.0284 ± 0.0010    |
| mpd-reimpl              | 100.00 ± 0.00  | 0.379 ± 0.008 | 98.86 ± 0.00            | 99.97 ± 0.00      | 1.7801 ± 0.0229    | 1.9351 ± 0.0271   | 1823.4021 ± 144.0529     | 0.0100 ± 0.0006    |
| mpd-splines-reimpl      | 100.00 ± 0.00  | 0.094 ± 0.002 | 93.25 ± 0.01            | 99.70 ± 0.00      | 1.7801 ± 0.0236    | 1.9411 ± 0.0281   | 48.9238 ± 5.2765         | 0.0090 ± 0.0006    |
| drmp (ours)             | 100.00 ± 0.00  | 0.016 ± 0.001 | 86.99 ± 0.02            | 99.44 ± 0.00      | 1.7900 ± 0.0236    | 1.9679 ± 0.0283   | 103.0449 ± 7.1893        | 0.0103 ± 0.0007    |

#### Extra objects

| algorithm               | success_rate   | time          | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:--------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 66.50 ± 5.50   | 3.528 ± 0.032 | 64.73 ± 0.05            | 99.09 ± 0.00      | 1.8416 ± 0.0285    | 1.8428 ± 0.0285   | 194.7415 ± 28.3457       | 0.0000 ± 0.0000    |
| grad                    | 34.67 ± 5.33   | 1.160 ± 0.008 | 34.67 ± 0.05            | 96.84 ± 0.00      | 1.9505 ± 0.0380    | 1.9509 ± 0.0382   | 39198.8847 ± 1232.7769   | 0.0000 ± 0.0000    |
| grad-splines            | 61.00 ± 5.67   | 1.188 ± 0.007 | 61.00 ± 0.05            | 99.21 ± 0.00      | 1.8550 ± 0.0307    | 1.8557 ± 0.0306   | 37.3256 ± 6.6706         | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 4.212 ± 0.264 | 98.01 ± 0.00            | 100.00 ± 0.00     | 2.1806 ± 0.0374    | 3.0325 ± 0.0455   | 866191.6630 ± 13439.4724 | 0.0381 ± 0.0010    |
| rrtconnect-spline       | 100.00 ± 0.00  | 2.653 ± 0.099 | 68.95 ± 0.01            | 99.24 ± 0.00      | 2.0376 ± 0.0319    | 2.6450 ± 0.0401   | 2915.7466 ± 232.0497     | 0.0400 ± 0.0012    |
| rrtconnect-gpmp2        | 100.00 ± 0.00  | 7.619 ± 0.379 | 99.92 ± 0.00            | 100.00 ± 0.00     | 1.9253 ± 0.0279    | 2.2691 ± 0.0370   | 40.1214 ± 3.1768         | 0.0279 ± 0.0010    |
| rrtconnect-grad         | 100.00 ± 0.00  | 4.710 ± 0.427 | 98.46 ± 0.00            | 99.99 ± 0.00      | 2.1836 ± 0.0298    | 2.7825 ± 0.0399   | 33766.7393 ± 348.9461    | 0.0416 ± 0.0011    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 4.691 ± 0.370 | 99.93 ± 0.00            | 100.00 ± 0.00     | 1.9513 ± 0.0287    | 2.3343 ± 0.0366   | 35.7235 ± 2.3047         | 0.0350 ± 0.0010    |
| mpd                     | 78.00 ± 4.67   | 0.483 ± 0.007 | 13.53 ± 0.02            | 97.83 ± 0.00      | 2.0764 ± 0.0442    | 2.3086 ± 0.0413   | 34903.0034 ± 4029.0827   | 0.0193 ± 0.0019    |
| mpd-reimpl              | 75.33 ± 5.00   | 0.384 ± 0.005 | 19.52 ± 0.02            | 96.02 ± 0.00      | 1.9448 ± 0.0326    | 2.0489 ± 0.0315   | 10279.8924 ± 699.8365    | 0.0067 ± 0.0008    |
| mpd-splines-reimpl      | 72.66 ± 5.00   | 0.094 ± 0.002 | 16.09 ± 0.02            | 93.72 ± 0.00      | 1.9384 ± 0.0347    | 2.0508 ± 0.0358   | 185.4848 ± 12.4824       | 0.0063 ± 0.0007    |
| drmp (ours)             | 73.33 ± 5.00   | 0.018 ± 0.001 | 14.60 ± 0.02            | 93.65 ± 0.00      | 1.9558 ± 0.0357    | 2.0708 ± 0.0359   | 271.4795 ± 14.5528       | 0.0067 ± 0.0007    |

### EnvDense2D

<img src="imgs/EnvDense2D.png" width="600">

#### Base

| algorithm               | success_rate   | time          | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:--------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 56.67 ± 5.67   | 2.953 ± 0.023 | 51.04 ± 0.06            | 98.70 ± 0.00      | 1.8292 ± 0.0299    | 1.8296 ± 0.0303   | 321.7846 ± 59.7824       | 0.0000 ± 0.0000    |
| grad                    | 20.83 ± 4.50   | 0.977 ± 0.003 | 20.83 ± 0.05            | 96.07 ± 0.00      | 1.9353 ± 0.0449    | 1.9362 ± 0.0453   | 39343.7655 ± 1915.5974   | 0.0000 ± 0.0000    |
| grad-splines            | 47.33 ± 5.67   | 1.036 ± 0.003 | 47.33 ± 0.06            | 98.79 ± 0.00      | 1.8310 ± 0.0343    | 1.8307 ± 0.0340   | 48.7880 ± 10.1116        | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 2.919 ± 0.145 | 97.54 ± 0.00            | 99.99 ± 0.00      | 2.2110 ± 0.0426    | 3.1077 ± 0.0522   | 880801.0992 ± 15760.4837 | 0.0413 ± 0.0013    |
| rrtconnect-spline       | 100.00 ± 0.00  | 2.761 ± 0.107 | 56.09 ± 0.01            | 99.03 ± 0.00      | 2.0583 ± 0.0380    | 2.6968 ± 0.0453   | 4340.0094 ± 397.5339     | 0.0440 ± 0.0014    |
| rrtconnect-gpmp2        | 100.00 ± 0.00  | 5.806 ± 0.110 | 98.64 ± 0.00            | 99.99 ± 0.00      | 1.9234 ± 0.0289    | 2.2805 ± 0.0370   | 60.4706 ± 4.2583         | 0.0327 ± 0.0010    |
| rrtconnect-grad         | 100.00 ± 0.00  | 4.164 ± 0.114 | 96.29 ± 0.00            | 99.97 ± 0.00      | 2.1950 ± 0.0334    | 2.7995 ± 0.0419   | 34494.6215 ± 264.6732    | 0.0440 ± 0.0012    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 4.799 ± 0.186 | 99.84 ± 0.00            | 100.00 ± 0.00     | 1.9486 ± 0.0295    | 2.3510 ± 0.0380   | 56.4135 ± 3.8072         | 0.0368 ± 0.0010    |
| mpd                     | 85.50 ± 3.83   | 0.428 ± 0.005 | 17.95 ± 0.02            | 98.15 ± 0.00      | 2.2695 ± 0.0441    | 2.4836 ± 0.0444   | 48994.6006 ± 3864.6963   | 0.0104 ± 0.0008    |
| mpd-reimpl              | 100.00 ± 0.00  | 0.279 ± 0.005 | 83.30 ± 0.01            | 99.65 ± 0.00      | 1.9971 ± 0.0323    | 2.4070 ± 0.0417   | 8902.2571 ± 309.9196     | 0.0337 ± 0.0011    |
| mpd-splines-reimpl      | 100.00 ± 0.00  | 0.100 ± 0.001 | 63.60 ± 0.02            | 98.17 ± 0.00      | 2.0032 ± 0.0357    | 2.3339 ± 0.0418   | 125.4364 ± 6.9123        | 0.0251 ± 0.0010    |
| drmp (ours)             | 100.00 ± 0.00  | 0.016 ± 0.001 | 44.70 ± 0.02            | 96.54 ± 0.00      | 2.0279 ± 0.0333    | 2.3608 ± 0.0380   | 272.8689 ± 10.6249       | 0.0290 ± 0.0011    |

#### Extra objects

| algorithm               | success_rate   | time           | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:---------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 43.00 ± 5.67   | 3.373 ± 0.045  | 34.78 ± 0.05            | 98.47 ± 0.00      | 1.8475 ± 0.0338    | 1.8484 ± 0.0337   | 562.9151 ± 90.1467       | 0.0000 ± 0.0000    |
| grad                    | 11.17 ± 3.50   | 1.888 ± 0.009  | 11.17 ± 0.04            | 95.23 ± 0.00      | 1.8990 ± 0.0619    | 1.8976 ± 0.0630   | 41783.8342 ± 2469.0239   | 0.0000 ± 0.0000    |
| grad-splines            | 30.50 ± 5.17   | 2.014 ± 0.010  | 29.64 ± 0.05            | 98.46 ± 0.00      | 1.8874 ± 0.0482    | 1.8899 ± 0.0487   | 83.7913 ± 25.1863        | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 6.899 ± 0.408  | 97.00 ± 0.00            | 99.99 ± 0.00      | 2.2951 ± 0.0395    | 3.2228 ± 0.0472   | 901428.9441 ± 12860.6889 | 0.0512 ± 0.0012    |
| rrtconnect-spline       | 100.00 ± 0.00  | 5.823 ± 0.261  | 45.18 ± 0.01            | 98.88 ± 0.00      | 2.1346 ± 0.0359    | 2.7575 ± 0.0435   | 8118.3568 ± 671.2641     | 0.0475 ± 0.0015    |
| rrtconnect-gpmp2        | 100.00 ± 0.00  | 10.869 ± 0.543 | 96.56 ± 0.00            | 99.98 ± 0.00      | 1.9740 ± 0.0272    | 2.3826 ± 0.0327   | 119.1933 ± 6.7214        | 0.0425 ± 0.0012    |
| rrtconnect-grad         | 100.00 ± 0.00  | 11.627 ± 0.650 | 89.47 ± 0.01            | 99.90 ± 0.00      | 2.2374 ± 0.0316    | 2.8261 ± 0.0363   | 35731.7650 ± 153.4618    | 0.0489 ± 0.0014    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 9.572 ± 0.859  | 99.19 ± 0.00            | 100.00 ± 0.00     | 2.0010 ± 0.0282    | 2.4551 ± 0.0345   | 103.0706 ± 6.3499        | 0.0440 ± 0.0012    |
| mpd                     | 86.83 ± 3.83   | 0.514 ± 0.003  | 15.80 ± 0.02            | 98.10 ± 0.00      | 2.3411 ± 0.0493    | 2.5388 ± 0.0465   | 60279.4413 ± 3377.2777   | 0.0107 ± 0.0008    |
| mpd-reimpl              | 98.83 ± 1.17   | 0.284 ± 0.004  | 29.84 ± 0.02            | 98.28 ± 0.00      | 2.0516 ± 0.0295    | 2.3423 ± 0.0316   | 16121.2264 ± 541.0624    | 0.0295 ± 0.0016    |
| mpd-splines-reimpl      | 98.17 ± 1.50   | 0.093 ± 0.002  | 19.81 ± 0.02            | 93.84 ± 0.00      | 2.0671 ± 0.0332    | 2.2965 ± 0.0346   | 243.6064 ± 10.8707       | 0.0194 ± 0.0015    |
| drmp (ours)             | 98.50 ± 1.17   | 0.018 ± 0.001  | 15.95 ± 0.01            | 92.78 ± 0.00      | 2.0841 ± 0.0314    | 2.2859 ± 0.0313   | 321.9948 ± 12.2196       | 0.0206 ± 0.0016    |

### EnvDenseNarrowPassage2D

<img src="imgs/EnvDenseNarrowPassage2D.png" width="600">

#### Base

| algorithm               | success_rate   | time           | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:---------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 35.33 ± 5.33   | 2.885 ± 0.008  | 35.33 ± 0.05            | 98.55 ± 0.00      | 1.8620 ± 0.0498    | 1.8615 ± 0.0501   | 116.7972 ± 17.4736       | 0.0000 ± 0.0000    |
| grad                    | 15.67 ± 4.00   | 1.067 ± 0.012  | 15.83 ± 0.04            | 95.43 ± 0.00      | 1.9512 ± 0.0658    | 1.9510 ± 0.0649   | 34986.7758 ± 1443.4563   | 0.0000 ± 0.0000    |
| grad-splines            | 36.00 ± 5.33   | 1.074 ± 0.011  | 35.83 ± 0.05            | 98.42 ± 0.00      | 1.8640 ± 0.0508    | 1.8640 ± 0.0504   | 10.2190 ± 2.0291         | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 8.130 ± 0.498  | 99.03 ± 0.00            | 100.00 ± 0.00     | 2.2339 ± 0.0400    | 3.0265 ± 0.0494   | 857212.2081 ± 14602.1589 | 0.0263 ± 0.0008    |
| rrtconnect-spline       | 100.00 ± 0.00  | 5.974 ± 0.424  | 71.95 ± 0.01            | 99.39 ± 0.00      | 2.0779 ± 0.0339    | 2.6440 ± 0.0422   | 2408.1137 ± 224.8312     | 0.0265 ± 0.0007    |
| rrtconnect-gpmp2        | 100.00 ± 0.00  | 10.078 ± 1.641 | 94.41 ± 0.02            | 99.99 ± 0.00      | 1.9402 ± 0.0280    | 2.1626 ± 0.0322   | 28.8911 ± 2.1500         | 0.0171 ± 0.0006    |
| rrtconnect-grad         | 100.00 ± 0.00  | 8.688 ± 1.792  | 93.98 ± 0.02            | 99.99 ± 0.00      | 2.1911 ± 0.0321    | 2.7419 ± 0.0385   | 33762.8483 ± 602.1265    | 0.0276 ± 0.0007    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 8.832 ± 1.939  | 95.10 ± 0.02            | 100.00 ± 0.00     | 1.9638 ± 0.0292    | 2.2559 ± 0.0326   | 31.7671 ± 2.8299         | 0.0205 ± 0.0006    |
| mpd                     | 98.17 ± 1.50   | 0.417 ± 0.003  | 90.44 ± 0.03            | 99.73 ± 0.00      | 1.9654 ± 0.0298    | 2.1459 ± 0.0303   | 4411.1169 ± 1429.7138    | 0.0117 ± 0.0005    |
| mpd-reimpl              | 99.50 ± 0.50   | 0.294 ± 0.005  | 92.94 ± 0.02            | 99.71 ± 0.00      | 1.9834 ± 0.0319    | 2.1629 ± 0.0324   | 3384.9187 ± 288.7196     | 0.0120 ± 0.0005    |
| mpd-splines-reimpl      | 98.83 ± 1.17   | 0.091 ± 0.002  | 88.45 ± 0.02            | 99.25 ± 0.00      | 1.9707 ± 0.0307    | 2.1446 ± 0.0311   | 68.8103 ± 5.7314         | 0.0106 ± 0.0004    |
| drmp (ours)             | 99.50 ± 0.50   | 0.016 ± 0.001  | 81.92 ± 0.02            | 98.82 ± 0.00      | 1.9756 ± 0.0311    | 2.1716 ± 0.0315   | 120.8311 ± 7.2181        | 0.0117 ± 0.0005    |

#### Extra objects

| algorithm               | success_rate   | time           | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                  | waypoints_stddev   |
|:------------------------|:---------------|:---------------|:------------------------|:------------------|:-------------------|:------------------|:-------------------------|:-------------------|
| gpmp2                   | 33.33 ± 5.33   | 3.390 ± 0.010  | 33.33 ± 0.05            | 97.51 ± 0.00      | 1.8710 ± 0.0399    | 1.8714 ± 0.0410   | 181.6919 ± 21.1022       | 0.0000 ± 0.0000    |
| grad                    | 12.83 ± 3.83   | 1.079 ± 0.002  | 12.83 ± 0.04            | 93.78 ± 0.00      | 2.0397 ± 0.0724    | 2.0378 ± 0.0722   | 41184.4318 ± 1607.2371   | 0.0000 ± 0.0000    |
| grad-splines            | 37.50 ± 5.50   | 1.370 ± 0.018  | 37.50 ± 0.05            | 97.67 ± 0.00      | 1.8809 ± 0.0405    | 1.8820 ± 0.0403   | 26.4692 ± 5.4555         | 0.0000 ± 0.0000    |
| rrtconnect              | 82.83 ± 4.17   | 18.398 ± 3.149 | 97.71 ± 0.00            | 99.99 ± 0.00      | 2.2459 ± 0.0401    | 2.9491 ± 0.0568   | 864134.5516 ± 17450.3788 | 0.0246 ± 0.0010    |
| rrtconnect-spline       | 82.83 ± 4.17   | 18.843 ± 3.235 | 60.11 ± 0.02            | 99.16 ± 0.00      | 2.0945 ± 0.0362    | 2.5793 ± 0.0484   | 4611.3917 ± 462.0190     | 0.0217 ± 0.0008    |
| rrtconnect-gpmp2        | 84.17 ± 4.17   | 19.654 ± 2.942 | 98.28 ± 0.01            | 100.00 ± 0.00     | 1.9154 ± 0.0296    | 2.1250 ± 0.0380   | 56.0280 ± 6.0765         | 0.0136 ± 0.0007    |
| rrtconnect-grad         | 84.17 ± 4.17   | 16.188 ± 2.552 | 94.80 ± 0.01            | 99.96 ± 0.00      | 2.1458 ± 0.0337    | 2.5997 ± 0.0439   | 34579.9367 ± 471.6204    | 0.0207 ± 0.0007    |
| rrtconnect-grad-splines | 84.17 ± 4.17   | 20.677 ± 3.171 | 98.63 ± 0.01            | 100.00 ± 0.00     | 1.9355 ± 0.0306    | 2.1792 ± 0.0388   | 46.4957 ± 5.3483         | 0.0157 ± 0.0006    |
| mpd                     | 82.33 ± 4.33   | 0.551 ± 0.003  | 50.57 ± 0.04            | 99.26 ± 0.00      | 1.9777 ± 0.0292    | 2.1478 ± 0.0297   | 24885.3383 ± 2462.0130   | 0.0100 ± 0.0005    |
| mpd-reimpl              | 82.00 ± 4.33   | 0.302 ± 0.004  | 32.22 ± 0.03            | 98.42 ± 0.00      | 1.9927 ± 0.0337    | 2.1183 ± 0.0358   | 12919.6824 ± 1031.1339   | 0.0078 ± 0.0006    |
| mpd-splines-reimpl      | 76.50 ± 4.83   | 0.095 ± 0.002  | 28.26 ± 0.03            | 95.62 ± 0.00      | 1.9960 ± 0.0370    | 2.0902 ± 0.0369   | 168.7034 ± 13.7743       | 0.0052 ± 0.0005    |
| drmp (ours)             | 76.83 ± 4.83   | 0.017 ± 0.001  | 27.00 ± 0.03            | 95.34 ± 0.00      | 2.0032 ± 0.0380    | 2.1122 ± 0.0361   | 237.7012 ± 15.2860       | 0.0061 ± 0.0006    |

## RobotL2D

L-shaped robot with width 0.3, height 0.4, and margin of 0.05.

### EnvSparse2D

<img src="imgs/EnvSparse2D.png" width="600">

#### Base

| algorithm               | success_rate   | time          | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                      | waypoints_stddev   |
|:------------------------|:---------------|:--------------|:------------------------|:------------------|:-------------------|:------------------|:-----------------------------|:-------------------|
| grad                    | 2.17 ± 1.50    | 4.312 ± 0.038 | 2.17 ± 0.01             | 56.29 ± 0.02      | 2.4043 ± 0.1397    | 2.4067 ± 0.1405   | 4033016.0000 ± 125282.4583   | 0.0000 ± 0.0000    |
| grad-splines            | 50.67 ± 5.67   | 4.361 ± 0.033 | 50.29 ± 0.06            | 98.15 ± 0.00      | 2.5565 ± 0.0908    | 2.5582 ± 0.0889   | 14015.4141 ± 448.6696        | 0.0000 ± 0.0000    |
| rrtconnect              | 100.00 ± 0.00  | 2.196 ± 0.407 | 99.98 ± 0.00            | 100.00 ± 0.00     | 2.7950 ± 0.0689    | 4.0540 ± 0.0990   | 41509113.1758 ± 1046877.6642 | 0.0326 ± 0.0019    |
| rrtconnect-spline       | 100.00 ± 0.00  | 2.262 ± 0.405 | 99.89 ± 0.00            | 99.99 ± 0.00      | 2.5575 ± 0.0604    | 3.5547 ± 0.0869   | 11367.8479 ± 1742.9271       | 0.0329 ± 0.0022    |
| rrtconnect-grad         | 100.00 ± 0.00  | 6.339 ± 0.456 | 99.89 ± 0.00            | 99.99 ± 0.00      | 2.9283 ± 0.0513    | 3.8244 ± 0.0760   | 2358923.6162 ± 47401.6670    | 0.0329 ± 0.0022    |
| rrtconnect-grad-splines | 100.00 ± 0.00  | 6.419 ± 0.448 | 99.67 ± 0.00            | 100.00 ± 0.00     | 2.6270 ± 0.0528    | 3.4871 ± 0.0720   | 13219.8087 ± 230.8026        | 0.0335 ± 0.0022    |
| mpd-reimpl              | 99.17 ± 0.83   | 0.366 ± 0.004 | 91.42 ± 0.02            | 99.52 ± 0.00      | 2.4962 ± 0.0539    | 3.0712 ± 0.0688   | 386862.0189 ± 34770.4653     | 0.0248 ± 0.0021    |
| mpd-splines-reimpl      | 98.83 ± 1.17   | 0.110 ± 0.002 | 76.61 ± 0.02            | 97.97 ± 0.00      | 2.4529 ± 0.0540    | 3.0056 ± 0.0687   | 10437.4670 ± 1524.5782       | 0.0219 ± 0.0018    |
| drmp (ours)             | 99.50 ± 0.50   | 0.036 ± 0.001 | 57.20 ± 0.02            | 95.53 ± 0.00      | 2.5306 ± 0.0566    | 3.1144 ± 0.0718   | 38365.3857 ± 7467.8262       | 0.0209 ± 0.0018    |

#### Extra objects
| algorithm               | success_rate   | time           | avg_free_trajectories   | avg_free_points   | path_length_best   | avg_path_length   | avg_ISJ                      | waypoints_stddev   |
|:------------------------|:---------------|:---------------|:------------------------|:------------------|:-------------------|:------------------|:-----------------------------|:-------------------|
| grad                    | 0.50 ± 0.50    | 5.563 ± 0.007  | 0.50 ± 0.01             | 37.77 ± 0.02      | -                  | -                 | -                            | -                  |
| grad-splines            | 5.17 ± 2.50    | 5.481 ± 0.005  | 5.17 ± 0.02             | 92.94 ± 0.00      | 2.3839 ± 0.1871    | 2.3882 ± 0.1889   | 13498.6925 ± 708.5607        | 0.0000 ± 0.0000    |
| rrtconnect              | 67.00 ± 5.33   | 61.041 ± 4.171 | 83.89 ± 0.03            | 99.20 ± 0.00      | 4.4084 ± 0.2002    | 5.8547 ± 0.2475   | 60311118.9774 ± 2427538.9040 | 0.0219 ± 0.0010    |
| rrtconnect-spline     | 67.00 ± 5.33   | 58.628 ± 3.488 | 73.81 ± 0.04            | 99.13 ± 0.00      | 4.1130 ± 0.1544    | 5.0678 ± 0.1772   | 2203384.8956 ± 60834.1580    | 0.0163 ± 0.0008    |
| rrtconnect-grad         | 67.00 ± 5.33   | 58.650 ± 3.460 | 78.59 ± 0.04            | 99.53 ± 0.00      | 3.6518 ± 0.1299    | 4.3976 ± 0.1397   | 18163.9567 ± 483.1098        | 0.0171 ± 0.0008    |
| rrtconnect-grad-splines | 67.00 ± 5.33   | 54.923 ± 3.583 | 83.60 ± 0.03            | 99.12 ± 0.00      | 3.9384 ± 0.1720    | 5.1115 ± 0.2140   | 687921.3311 ± 133986.4331    | 0.0175 ± 0.0008    |
| mpd-reimpl              | 12.17 ± 3.50   | 0.353 ± 0.002  | 0.76 ± 0.00             | 80.43 ± 0.01      | 3.1451 ± 0.2299    | 3.2760 ± 0.2018   | 1726135.0683 ± 620712.6040   | 0.0019 ± 0.0010    |
| mpd-splines-reimpl      | 3.83 ± 2.17    | 0.126 ± 0.002  | 0.17 ± 0.00             | 55.07 ± 0.01      | 2.3777 ± 0.2018    | 2.4924 ± 0.2115   | 7792.0549 ± 3639.7241        | 0.0007 ± 0.0004    |
| drmp (ours)             | 3.33 ± 2.00    | 0.047 ± 0.001  | 0.08 ± 0.00             | 51.52 ± 0.01      | 2.5525 ± 0.2726    | 2.6041 ± 0.2568   | 11639.9039 ± 4108.3258       | 0.0004 ± 0.0004    |
| drmp-grad-splines       | 15.83 ± 4.17   | 5.977 ± 0.046  | 0.50 ± 0.00             | 88.71 ± 0.00      | 2.8716 ± 0.1408    | 2.9247 ± 0.1337   | 9968.3573 ± 539.3666         | 0.0009 ± 0.0004    |

---
<img src="imgs/task0-trajectories-DRMP.png" width="400"> <img src="imgs/task0-trajectories-MPD-Splines.png" width="400"> 
<img src="imgs/task0-trajectories-DRMP-extra.png" width="400"> <img src="imgs/task0-trajectories-MPD-Splines-extra.png" width="400"> 

Comparison of DRMP on the left (1-step generation) vs MPD-Splines-reimpl on the right (32 steps generation). Shortcut training doesn't reduce the diversity of trajectories learned by the model.

## 4. Limitations and Future Work
One limitation of the proposed approach is observed in the experiment with an L-shaped robot. The algorithm struggles to generalize to environments where extra obstacles introduce significant challenges to trajectory generation that cannot be solved using simple optimization. One possible research direction addressing this issue involves designing a model that takes the environment as input and encodes it using an extra set of weights. However, training such models is non-trivial, especially regarding data generation.

Another interesting research direction (connected to the first) is to learn trajectory representations in the latent space of a CVAE rather than in the space of B-Spline control points. Such a shift is promising as it would enable the application of novel generative model training paradigms, such as flow matching or drifting models. Additionally, learning to encode map information would be easier with latent representations of both the map and trajectories.

## 5. Conclusion
We presented DRMP, a diffusion-based motion planner optimized for speed. By leveraging shortcut models, we compressed the iterative diffusion process into a single step. Our experiments demonstrate that DRMP provides a high-quality initial seed for motion planning problems in a fraction of the time required by standard diffusion models or sampling-based planners, paving the way for real-time learned motion planning.

## References
[1] Carvalho, J., et al. "Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models." arXiv preprint arXiv:2412.19948 (2024).
[2] Liu, K., et al. "One Step Diffusion via Shortcut Models." arXiv preprint arXiv:2410.12557 (2024).
