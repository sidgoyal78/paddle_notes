### Merging all params in a single file


#### Understanding save/load ops (C++ side)
- From the [model_format design doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/model_format.md), we see some details in the table but it is not super clear. So we will look at the implementation.


To understand the current serialization: we look at `save_op`

- In `save_op` the main work is performed by `SerializeToStream( <ofstream>, <framework::LoDTensor>, .. )` [Code](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.cc#L235). This function saves a version number, size of LoD and actual LoD data. 

- Then it calls, `SerializeToStream(<ofstream>, <Tensor> ..)` [Code](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/tensor_util.h#L218). This function saves a version number, tensor description as a serialized protobuf, and the actual data.  

The corresponding `load_op` basically does the deserialization accordingly (respecting the ordering in the `save_op`).


#### Understanding how a model is saved (python api)
Now, we look at how the save/load works for saving actual model params, we look at the implementation of `save_vars` in fluid. [Code](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/io.py#L88). We see that a new program is created with `save` op is appended for each `vars` which is persistable. Then the executor runs this program.

#### Approach

We basically make two assumptions:
- For both load/save, the order of iterating over the variables is the same. (This should hopefully be true)
- We don't worry about the `overwrite` option which is in `save_op`.

While saving:
- We basically store a `uint64_t` number in addition to the actual serialized bytes as in the original `save`. This number will tell us about the size of the serialized LoDTensor in bytes.

- When the `save` is called for the first time, we will create a file, create a string that will have serialized LoDTensor data. Now we store the size of this string first in a fixed width (`uint64_t`) number, and then store the string.

- When the `save` is called later, we basically go to the end of the file, and store 2 things: the size of the string and the string itself.


While loading:
- We pass an additional attribute, in order to load the correct chunk of parameter. So we pass a counter value (which counts from 0 the relative order of the different params). 

- With this counter and the extra size information that we stored, we can hop to the appropriate part of the file, and read the chunk, and deserialize it.


