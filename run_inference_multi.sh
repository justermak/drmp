#!/bin/bash

# Define the list of algorithms
algorithms=(
    "generative-model"
    "mpd"
    "mpd-splines"
    "rrt"
    "rrt-smooth"
    "gpmp2"
    "grad"
    "rrt-gpmp2"
    "rrt-grad"
    "rrt-grad-splines"
)

for algo in "${algorithms[@]}"; do
    echo "Running inference for $algo..."
    python3 scripts/inference.py --algorithm "$algo"
done
