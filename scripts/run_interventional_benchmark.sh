#!/bin/bash

# Define an array of values for d
declare -a d_vals=(5 10 25 50 100)

# Define an array of values for edge density
declare -a p_vals=(.05 .1)

# Define an array of values for seed
declare -a seed_vals=(0 1 2)

# Loop through each combination of d and seed values
for d in "${d_vals[@]}"
do
    for p in "${p_vals[@]}"
    do
        for seed in "${seed_vals[@]}"
        do
            # Run the Python script with the current values of d and seed
            python3 scripts/interventional_benchmark.py --n 50 --d $d --p $p --seed $seed --model SDCI --force True

            # Add a newline for separation between iterations
            echo ""
        done
    done
done

