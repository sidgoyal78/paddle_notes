### Goal: To benchmark inference API unit tests for all chapters

- Need numbers for both CPU and GPU
- Need to distinguish time taken for data preparation and time to actually run
- Need to get numbers for varying batch sizes


|    mb=1 |    mb=2 |    mb=8 |    mb=32 |    mb=64 |   mb=128 |   mb=256 |
|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| 1.29664 | 2.30658 | 3.84233 | 10.4951  | 20.8305  |  26.2213 |  49.6337 |
| 2.66379 | 2.81712 | 2.85527 |  2.39371 |  2.39629 |   2.1023 |   2.4629 |
