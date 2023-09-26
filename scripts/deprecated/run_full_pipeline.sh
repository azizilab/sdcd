#!/bin/bash

# Define an array of values for d
# declare -a d_vals=(20 50 100 200)
declare -a d_vals=(20)

# Define an array of values for seed
declare -a seed_vals=(0 1 2)

# Loop through each combination of d and seed values
for d in "${d_vals[@]}"
do
    for seed in "${seed_vals[@]}"
    do
        # Run the Python script with the current values of d and seed
	python3 full_pipeline_main.py --d $d --seed $seed --model dcdi

        # Add a newline for separation between iterations
        echo ""
    done
done
