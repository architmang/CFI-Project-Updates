#!/bin/bash

# Set the input and output directories
input_directory="archit_color"
output_directory="archit_color_openpose"

# Iterate over the images in the input directory
for image_file in "${input_directory}"/*.jpg; do
  # Get the image name without the directory path
  image_name=$(basename "${image_file}")
  
  # Set the output path for the processed image
  output_path="${output_directory}/${image_name}"
  
  # Perform pose estimation using OpenPose
  ./openpose/build/examples/openpose/openpose.bin --image_dir "${input_directory}" --write_images "${output_directory}"
  
  echo "Processed image: ${image_file}"
done

bin\OpenPoseDemo.exe --image_dir archit_color --write_images archit_color_openpose