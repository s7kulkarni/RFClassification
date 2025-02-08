#!/bin/bash

# Set the base folder paths
SOURCE_DIR="DroneRF"
DEST_DIR="DroneRF_extracted"

# Check if the source folder exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source folder $SOURCE_DIR does not exist."
  exit 1
fi

# Create the destination folder if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
  mkdir "$DEST_DIR"
fi

# Loop through the subfolders in the source directory
for subfolder in "$SOURCE_DIR"/*; do
  if [ -d "$subfolder" ]; then
    # Get the subfolder name
    subfolder_name=$(basename "$subfolder")
    
    # Create corresponding subfolder in destination directory
    dest_subfolder="$DEST_DIR/$subfolder_name"
    if [ ! -d "$dest_subfolder" ]; then
      mkdir "$dest_subfolder"
    fi
    
    # Extract all .rar files in the subfolder to the corresponding subfolder in destination
    for rar_file in "$subfolder"/*.rar; do
      if [ -f "$rar_file" ]; then
        echo "Extracting $rar_file to $dest_subfolder"
        unar -o "$rar_file" "$dest_subfolder/"
      fi
    done
  fi
done

echo "Extraction completed!"