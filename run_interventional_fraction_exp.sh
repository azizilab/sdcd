#!/bin/bash
# This script evaluate the performance of the models on a varying fraction of interventional data

# Define an array of values for the fraction of interventional data
declare -a frac_vals=(0.0 0.25 0.5 0.75 1.0)

# Define an array of values for d
declare -a d_vals=(20 50)

# Define an array of values for seed
declare -a seed_vals=(0)

# Loop through each combination of frac, d, and seed values
for frac in "${frac_vals[@]}"
do
    for d in "${d_vals[@]}"
    do
        for seed in "${seed_vals[@]}"
        do
          echo "Running with frac=$frac, d=$d, seed=$seed"
            # Run the Python script with the current values of frac, d, and seed
      python3 full_pipeline_main.py --d $d --seed $seed --model sdci --frac_interventions $frac --n 100 \
                                    --wandb-project sdci_interventional_fraction

            # Add a newline for separation between iterations
            echo ""
        done
    done
done