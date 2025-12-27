## 1. Data Generation 

| Strategy | Ease of Implementations | Potential | Description |
| :--- | :---: | :---: | :--- |
| **Trajectory slicing** | 5 | 5 | Cut slices of trajectories, smoothen and increase the n_support_points for the slice using B-spline |
| **Time reversal** | 5 | 5 | Swap start and goal and reverse the trajectory |
| **Start/Goal jittering** | 4 | 2 | Move start/goal a little, fix trajectory by optimizing a little |
| **Labeling samples** | 3 | 3 | Different labels for shortest and safest trajectories |
| **Normalization** | 5 | 3 | Try $N(0, 1)$ instead of $[-1; 1]$ |

Add more metadata to trajectories: trajectory length, sharpness, costs, min distance to obstacle

---

## 2. Training 

| Strategy | Ease of Implementations | Potential | Description |
| :--- | :---: | :---: | :--- |
| **Conditioning ideas** | 5 | ??? | Add length / cost / cls token to context |
| **Trajectory smoothening** | 4 | 5 | Apply B-spline smoothening to the model's outputs before backward |
| **Predict $x_0$** | 5 | 3 | Predicting the trajectory instead of the noise |
| **Classifier-Free Guidance** | 2 | 3 | Randomly drop the goal condition during training |

---

## 3. Architecture

1. Positional encodings - random fourier vs learned fourier vs default sinusoidal
2. Attention or no attention
3. Numbers

---

## 4. Inference

1. Gradient based guidance
2. Manifold projection

---

## 5. Moving forward

1. Shortcut model
2. Energy based model contrastive training
