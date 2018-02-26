## Task

We benchmark the inference unit-test corresponding to the "recognize_digits" example. Basically, the architecture consists of conv2d, pool2d, and fc layers.


## Hardware specification
We use a machine with the following specification:
- CPU: Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz
- GPU: GeForce GTX 1080 Ti GPU card.
- Code: PR https://github.com/PaddlePaddle/Paddle/pull/8497


## Total time

We first look at the total time for doing inference for both CPU and GPU, with different batch-sizes. The important point to note here are: 
- For GPU timings, only the operators run on GPU and for the total time we use the CPU time. We look at individual time of ops later.
- We don't consider the time to setup the inputs, however, we have the data for that separately.
- All times are in ms (millisecond)
- Each value presented below, represents the average value averaged over 10 independent runs.


|   |    mb=1 |    mb=2 |    mb=8 |    mb=32 |    mb=64 |   mb=128 |   mb=256 |
|---|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
|CPU| 1.29664 | 2.30658 | 3.84233 | 10.4951  | 20.8305  |  26.2213 |  49.6337 |
|GPU| 2.66379 | 2.81712 | 2.85527 |  2.39371 |  2.39629 |   2.1023 |   2.4629 |


## Time for individual ops

We then look at the time for individual ops and compare both CPU and GPU, for each minibatch size.

- For batchsize = 1

|     |    conv2d |       Relu |    pool2d |   batchnorm |   conv2d |      Relu |    pool2d |      mul: |   total |
|:----|----------:|-----------:|----------:|------------:|---------:|----------:|----------:|----------:|--------:|
| CPU | 0.0984842 | 0.00849511 | 0.0287931 |   0.0117988 | 0.164882 | 0.005549  | 0.0119113 | 0.011454  | 1.29664 |
| GPU | 0.105013  | 0.0156029  | 0.0303324 |   0.0560818 | 0.107438 | 0.0153963 | 0.0294684 | 0.0237003 | 2.66379 |

- For batchsize = 2

|     |   conv2d |      Relu |    pool2d |   batchnorm |   conv2d |      Relu |    pool2d |      mul: |   total |
|:----|---------:|----------:|----------:|------------:|---------:|----------:|----------:|----------:|--------:|
| CPU | 0.266398 | 0.0153681 | 0.0852202 |   0.0202848 | 0.423694 | 0.0102443 | 0.0303461 | 0.0170684 | 2.30658 |
| GPU | 0.113596 | 0.0210927 | 0.0420587 |   0.070432  | 0.124075 | 0.0189159 | 0.0407751 | 0.032861  | 2.81712 |

- For batchsize = 8

|     |   conv2d |      Relu |    pool2d |   batchnorm |   conv2d |      Relu |    pool2d |      mul: |   total |
|:----|---------:|----------:|----------:|------------:|---------:|----------:|----------:|----------:|--------:|
| CPU | 0.532441 | 0.0605997 | 0.281486  |    0.029859 | 1.16901  | 0.0191726 | 0.0952213 | 0.0248031 | 3.84233 |
| GPU | 0.087904 | 0.0225621 | 0.0449173 |    0.077216 | 0.123534 | 0.020052  | 0.0429867 | 0.0398079 | 2.85527 |

- For batchsize = 32

|     |    conv2d |      Relu |    pool2d |   batchnorm |   conv2d |      Relu |   pool2d |      mul: |    total |
|:----|----------:|----------:|----------:|------------:|---------:|----------:|---------:|----------:|---------:|
| CPU | 2.25282   | 0.200443  | 0.948266  |   0.063136  | 4.77001  | 0.0558134 | 0.279895 | 0.0538337 | 10.4951  |
| GPU | 0.0733227 | 0.0183406 | 0.0396693 |   0.0642809 | 0.135627 | 0.0183451 | 0.036384 | 0.0319942 |  2.39371 |

- For batchsize = 64

|     |    conv2d |      Relu |    pool2d |   batchnorm |   conv2d |      Relu |   pool2d |      mul: |    total |
|:----|----------:|----------:|----------:|------------:|---------:|----------:|---------:|----------:|---------:|
| CPU | 4.45281   | 0.328914  | 2.44761   |   0.110637  | 9.47443  | 0.08905   | 0.716909 | 0.0407523 | 20.8305  |
| GPU | 0.0878791 | 0.0178758 | 0.0459484 |   0.0657458 | 0.134663 | 0.0165089 | 0.033536 | 0.0306724 |  2.39629 |

- For batchsize = 128

|     |   conv2d |      Relu |    pool2d |   batchnorm |    conv2d |      Relu |    pool2d |      mul: |   total |
|:----|---------:|----------:|----------:|------------:|----------:|----------:|----------:|----------:|--------:|
| CPU | 4.76558  | 0.861102  | 2.94608   |   0.219243  | 12.6843   | 0.235088  | 0.911418  | 0.0612628 | 26.2213 |
| GPU | 0.100658 | 0.0165576 | 0.0370809 |   0.0686613 |  0.140516 | 0.0129161 | 0.0320249 | 0.0296862 |  2.1023 |

- For batchsize = 256

|     |   conv2d |      Relu |    pool2d |   batchnorm |    conv2d |      Relu |    pool2d |      mul: |   total |
|:----|---------:|----------:|----------:|------------:|----------:|----------:|----------:|----------:|--------:|
| CPU |  9.25957 | 2.31101   | 5.48147   |   0.509528  | 24.1498   | 0.544391  | 1.74179   | 0.08145   | 49.6337 |
| GPU |  0.12006 | 0.0247442 | 0.0504711 |   0.0744107 |  0.225806 | 0.0187989 | 0.0387449 | 0.0334994 |  2.4629 |

This probably suggests that the CPU kernels of both pool2d and conv2d ops could be inspected further to see if we can optimize.


Note: The scripts for getting these numbers and the log files can be found in the `data/` folder.
