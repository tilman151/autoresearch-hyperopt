#!/bin/bash

# Run Random sampler for 12 hours
echo "Starting Random sampler optimization for 12 hours..."
uv run train.py --sampler Random --timeout 12 --n-trials 0 --study-name hyperopt_random

# Run GP sampler for 12 hours
echo "Starting GP sampler optimization for 12 hours..."
uv run train.py --sampler GP --timeout 12 --n-trials 0 --study-name hyperopt_gp

# Run CMAES sampler for 12 hours
echo "Starting CMAES sampler optimization for 12 hours..."
uv run train.py --sampler CMAES --timeout 12 --n-trials 0 --study-name hyperopt_cmaes

echo "All hyperparameter optimization studies completed."
