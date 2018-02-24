
## Problem description

The main goal is to come up with an approach for integrating TensorRT with PaddlePaddle's inference library. We want to do this in order to use TensorRT for performing inference on a model saved using Fluid.

To address this, we will first briefly discuss TensorRT, and the functionalities offered by TensorRT, before finally concluding.

## TensorRT 

#### Introduction

TensorRT is deep learning inference optimizer and runtime from Nvidia, aimed at deploying trained deep networks for inference in a variety of production platforms. Using TensorRT involves two phases:
- Build phase: In the build phase, TensorRT performs optimizations on the configuration of the neural network and generates an optimized *plan* for the forward pass computation.
- Deployment phase: In this phase, TensorRT performs inference by executing the *plan* on the input data (which comes from a services or a user application).

#### Build phase

This step is performed only once, prior to deployment. A "trained model" tranined using any popular deep learning framework has to be first parsed using TensorRT, and imported to the TensorRT Optimizer module. The TensorRT Optimizer performs several optimizations (briefly discussed below) and outputs an optimized inference execution engine. This execution engine when serialized to a file on disk is known as *plan* file.

The crucial part here is importing a trained model. For Caffe and Tensorflow, TensorRT provides simple Python and C++ APIs to import the models directly. However, for other frameworks, we need to use TensorRT's Network Definition API to specify the network description (either in C++ or Python), before loading it into TensorRT.

An image summarizing this phase is: https://devblogs.nvidia.com/wp-content/uploads/2017/12/pasted-image-0-4-768x656.png


The various optimizations performed by the TensorRT Optimizer are:
- Graph optimizations are performed to restructure the graph by doing layer and tensor fusion.
- FP16 and INT8 percision caliberation is supported to convert FP32 to lower precision.
- Kernel auto tuning is performed to choose the best implementation of kernels from a library of kernels, for the given input data size, layout, etc.
- Reduction of memory footprint, by reusing memory for each tensor.

#### Deploy phase

In this phase, the saved *plan* file is loaded and deserialized to create a TensorRT Runtime engine object and used to perform inference on new data.


## Our approach

As discussed in the "build phase" subsection, the most important point for our usecase is: to import a model into TensorRT that is trained using PaddlePaddle fluid. 

From the documentation, we find that networks from other frameworks (except Caffe and Tensorflow) via the UFF format. The UFF: Universal Framework Format is a data format that describes an execution graph for a deep network.
The format consists of syntax for serialization format and definition of each operators (as protobuf and python descriptors respectively).

The documentation contains an example of using TensorRT's Python API to convert a model from PyTorch into a TensorRT engine. However, there isn't any example demonstrating TensorRT's C++ API to convert a model from any other framework. So the first task is to come up with an example where we can use TensorRT's C++ API to convert a model to the required format.

Regarding current support of ONNX with TensorRT: 
- The developer's guide doesn't discuss about ONNX, hence it seems that only Caffe and TensorFlow converters are provided.
- Looking at the blog post: https://devblogs.nvidia.com/tensorrt-container/, and actually inspecting the code inside the docker container, we don't see any source files. 

Thus, we think it is reasonable to come up with a custom converter that imports fluid's model into TensorRT (using the C++ API).

## References:

- TensorRT 3.0 developer guide: http://docs.nvidia.com/deeplearning/sdk/pdf/TensorRT-Developer-Guide.pdf
- TensorRT blog discussing the optimizations in build phase: https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/
- TensorRT blogpost: https://devblogs.nvidia.com/deploying-deep-learning-nvidia-tensorrt/
