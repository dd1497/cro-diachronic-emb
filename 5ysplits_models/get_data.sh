#!/bin/bash

# Download the zip file
echo "Downloading cro-diachronic-emb.zip..."
wget https://www.takelab.fer.hr/cro-diachronic-emb.zip

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download the file"
    exit 1
fi

# Unpack the zip file
echo "Unpacking archive..."
unzip cro-diachronic-emb.zip

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to unpack the archive"
    exit 1
fi

# Move contents out of any subdirectory to current directory
# This handles cases where files are in a subdirectory within the zip
if [ -d "cro-diachronic-emb" ]; then
    echo "Moving files from subdirectory..."
    mv cro-diachronic-emb/* .
    rmdir cro-diachronic-emb
fi

# Remove the original zip file
echo "Cleaning up..."
rm cro-diachronic-emb.zip

echo "Done! Files are now in the current directory."

