

## Input protobuf for Inference Engine

Please refer to Xreki's excellent PR [#7315](https://github.com/PaddlePaddle/Paddle/pull/7315) to get an overview of the design of Inference Engine.

The design of inference engine depends on:
> how we store the protobuf message of the ProgramDesc of the `main_program` in fluid. 

The aim of this document is to look at two different approaches for doing it, and evaluate some of the pros and cons of each approach.

If we look at an existing training and inference example in fluid for example [recognize_digits](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/tests/book/test_recognize_digits_mlp.py), we see that there are two objects of the [Program class](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/framework.py#L786) in Python, each of which has a ProgramDesc as its member. 

The ProgramDesc obtained from the `default_main_program()` basically represents the neural network during the training phase. However, for testing/inference phase, as in the above example, we `prune()` and `inference_optimize()` the original ProgramDesc, to prune out irrelevant operators and obtain a ProgramDesc (in the `inference_program`) which is suitable for inference. (For more details, see implementation: [link](https://github.com/PaddlePaddle/Paddle/blob/e445b3ff20f0c568b7d01ed91cbd154c745e124c/paddle/framework/prune.cc)). To simplify the writing, we call the first ProgramDesc as "training ProgramDesc" and the latter one as "inference ProgramDesc". An important thing to note is that "inference ProgramDesc" has lesser information that the "training ProgramDesc" as we prune out operators which aren't required for inference.

### Approach 1: Use inference ProgramDesc 
Under the subsection [inference-program](https://github.com/Xreki/Paddle/blob/acd813127c13f94bf98372215c96f5cf676a649c/doc/design/inference.md#inference-program), it is proposed that the "protobof message of the `main_program` is saved using `fluid.io.save_inference_model`." 

Based on the current implementation of `fluid.io.save_inference_model`, we observe that "inference ProgramDesc" is stored (in addition to model parameters). Now, there are again two options:

1. We can modify the current implementation to save the inference ProgramDesc which will have `feed` and `fetch` operators as well (which isn't done in the current implementation). This has the benefit that the user who wants to do inference, doesn't need to worry about providing feed and fetch lists to the Inference engine API.

2. We use the current implmentation as is, and save the inference ProgramDesc without `feed` and `fetch` operators. Here, the user must provide feed and fetch lists as input to the Inference engine API.

However, the main drawback of both of the above options and saving inference ProgramDesc is that we will need to have extra provisions to allow online training. More specifically, we will need to also save the "training ProgramDesc" and read it additionally to support online training. Moreover, we might face issues related to parameter sharing  when we want to do both inference and also modify the parameters. 

### Approach 2: Use training ProgramDesc 

To address the limitation of the first approach, at the cost of extra computation, is to save the "training ProgramDesc".  In order to do inference, we can perform pruning and optimization on this ProgramDesc to obtain the "inference ProgramDesc". However, in order to do the pruning, we will need the user to specify feed and fetch lists to the Inference engine.

Even though we will have to perform pruning on the "training ProgramDesc" in the Inference Engine, we will still be able to support online training, as we won't have to worry about saving/reading an additional "training ProgramDesc". 
