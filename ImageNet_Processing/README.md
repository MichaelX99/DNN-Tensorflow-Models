1- Download the Imagenet2012 classification dataset and the dev kit and untar the train and validation file.
2- Copy the create.sh script into the train directory (chmod +x if needed), and run the script
3- Run transform.py to get the dataset into order and create the label.txt file

The rest is taken from the Inception tutorial on github @ https://github.com/tensorflow/models/tree/master/inception

4- Set TRAIN_DIR equal to the top level of your training data is
5- Set VALIDATION_DIR equal to the top level of your evaluation data
6- Set OUTPUT_DIRECTORY equal to where you want your TFRecord files to go (must create the directory if necessary)
7- Set LABELS_FILE equal to where your labels file is
8- bazel build //ImageNet:build_image_data
9- bazel-bin/ImageNet/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
