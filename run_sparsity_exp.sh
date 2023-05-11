#!/bin/bash

declare -a n_edges_per_d_vals=(2 5 10 20)

# Define an array of values for seed
declare -a seed_vals=(0)

# Loop through each combination of d and seed values
for nd in "${n_edges_per_d_vals[@]}"
do
    for seed in "${seed_vals[@]}"
    do
        # Run the Python script with the current values of d and seed
    python3 full_pipeline_main.py --d 50 --n_edges_per_d $nd --seed $seed --model sdcd

        # Add a newline for separation between iterations
        echo ""
    done
done


for nd in "${n_edges_per_d_vals[@]}"
do
    for seed in "${seed_vals[@]}"
    do
        # Run the Python script with the current values of d and seed
    python3 full_pipeline_main.py --d 50 --n_edges_per_d $nd --seed $seed --model dcdfg

        # Add a newline for separation between iterations
        echo ""
    done
done