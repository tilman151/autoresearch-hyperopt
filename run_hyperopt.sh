#!/bin/bash

# Run Random sampler for 8 hours
echo "Starting Random sampler optimization for 8 hours..."
uv run train.py --sampler Random --timeout 8 --n-trials 0 --study-name hyperopt_random

# Run GP sampler for 8 hours
echo "Starting GP sampler optimization for 8 hours..."
uv run train.py --sampler GP --timeout 8 --n-trials 0 --study-name hyperopt_gp

# Run CMAES sampler for 8 hours
echo "Starting CMAES sampler optimization for 8 hours..."
uv run train.py --sampler CMAES --timeout 8 --n-trials 0 --study-name hyperopt_cmaes

echo "All hyperparameter optimization studies completed."
