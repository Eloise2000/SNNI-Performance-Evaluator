#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="./script/avgbn_create_model.py"

# Define an array of random seeds
RANDOM_SEEDS=(17 23 36 42 49 53 62 79 81 97)

# Define the folder path you want to pass to the Python script
FOLDER_PATH="./models/"

# Define whether the built model being tested after creation
TEST_FLAG="" # Set to "True" if test, else ""

# Loop through each random seed and run the Python script
for SEED in "${RANDOM_SEEDS[@]}"; do
    echo -e "********** Running convmp script with random seed $SEED **********"
    python $SCRIPT_PATH --random_seed $SEED --folderpath $FOLDER_PATH --test "$TEST_FLAG"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run script with random seed $SEED."
        exit 1
    fi
    echo -e "\nFinished running script with random seed $SEED.\n"
done

echo "All random seeds processed successfully."