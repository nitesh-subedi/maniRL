#!/bin/bash

# Define base log directory
base_name="her_reward_v3"

# Specify the full path to your Python interpreter
python_interpreter="/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh"

# Number of iterations
iterations=100

# Initialize the variable to store the last saved model path
last_saved_model=""
last_saved_model_replay_buffer=""

# Loop to run the Python program multiple times
for ((i=1; i<=iterations; i++))
do
    # Create a unique run name for each iteration
    name="${base_name}_run_${i}"
    
    # Run the Python script, passing the run name and the last saved model (if any)
    if [ -z "$last_saved_model" ]; then
        # No model available, run without loading a model
        $python_interpreter train.py --run_name $name
    else
        # Pass the last saved model to the Python script
        $python_interpreter train.py --run_name $name --load_model $last_saved_model --load_replay_buffer $last_saved_model_replay_buffer
    fi
    
    # Update the last saved model path (assumes the model is saved as mybuddy_policy_checkpoint.zip)
    last_saved_model="./real_env_results/${name}/mybuddy_policy_checkpoint_10000_steps.zip"
    last_saved_model_replay_buffer="./real_env_results/${name}/replay_buffer.pkl"
done
