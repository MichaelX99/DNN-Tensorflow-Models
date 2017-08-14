# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//ImageNet/..."],
)

py_binary(
    name = "build_image_data",
    srcs = ["data/build_image_data.py"],
)
