# Functionalities
## 1. Model Profiling
### Basic profiling

- Kindle model provides profiling option.
- Please refer to [API reference page](https://limjk.ai/kindle/api/kindle.utils.model_utils/#kindleutilsmodel_utilsmodelprofiler) for the detailed information.

```python
from kindle import Model

model = Model("model.yaml")

profiler = model.profile(n_run=100, batch_size=32, input_size=(224, 224), verbose=True)
```

- Above code will print the profiling result as following.

```shell
Profiling result by 100 times running. Sorted by running order.
------------------------------------------------------------------------------------------------------
 idx |                 Name | Time(Mean) |  Time(Std) | Time(Total) | Rank |   Ratio |        Params |
------------------------------------------------------------------------------------------------------
   0 |                 Conv |    1.40 ms |  566.86 μs |   139.69 ms |    1 |  36.08% |           616 |
   1 |              MaxPool |  410.78 μs |  136.45 μs |    41.08 ms |    3 |  10.61% |             0 |
   2 |            nn.Conv2d |  980.33 μs |  459.86 μs |    98.03 ms |    2 |  25.32% |         3,200 |
   3 |       nn.BatchNorm2d |  277.48 μs |  141.98 μs |    27.75 ms |    5 |   7.17% |            32 |
   4 |              nn.ReLU |  109.23 μs |   43.48 μs |    10.92 ms |    7 |   2.82% |             0 |
   5 |              MaxPool |  242.73 μs |   80.28 μs |    24.27 ms |    6 |   6.27% |             0 |
   6 |              Flatten |   18.18 μs |   16.98 μs |     1.82 ms |   10 |   0.47% |             0 |
   7 |               Linear |  294.72 μs |  192.97 μs |    29.47 ms |    4 |   7.61% |       123,000 |
   8 |               Linear |   94.48 μs |   44.85 μs |     9.45 ms |    8 |   2.44% |        10,164 |
   9 |               Linear |   46.44 μs |   26.23 μs |     4.64 ms |    9 |   1.20% |           850 |
------------------------------------------------------------------------------------------------------
Running time
 - Total :   387.13 ms
 -  Mean :     3.87 ms
 -   STD :     1.45 ms
```

### Profiling by time consumption order

- Profiling result can be sorted by running times.

```python
profiler.print_result(sort_by_rank=True)
```

```shell
Profiling result by 100 times running. Sorted by time consumption.
------------------------------------------------------------------------------------------------------
 idx |                 Name | Time(Mean) |  Time(Std) | Time(Total) | Rank |   Ratio |        Params |
------------------------------------------------------------------------------------------------------
   0 |                 Conv |    1.40 ms |  566.86 μs |   139.69 ms |    1 |  36.08% |           616 |
   2 |            nn.Conv2d |  980.33 μs |  459.86 μs |    98.03 ms |    2 |  25.32% |         3,200 |
   1 |              MaxPool |  410.78 μs |  136.45 μs |    41.08 ms |    3 |  10.61% |             0 |
   7 |               Linear |  294.72 μs |  192.97 μs |    29.47 ms |    4 |   7.61% |       123,000 |
   3 |       nn.BatchNorm2d |  277.48 μs |  141.98 μs |    27.75 ms |    5 |   7.17% |            32 |
   5 |              MaxPool |  242.73 μs |   80.28 μs |    24.27 ms |    6 |   6.27% |             0 |
   4 |              nn.ReLU |  109.23 μs |   43.48 μs |    10.92 ms |    7 |   2.82% |             0 |
   8 |               Linear |   94.48 μs |   44.85 μs |     9.45 ms |    8 |   2.44% |        10,164 |
   9 |               Linear |   46.44 μs |   26.23 μs |     4.64 ms |    9 |   1.20% |           850 |
   6 |              Flatten |   18.18 μs |   16.98 μs |     1.82 ms |   10 |   0.47% |             0 |
------------------------------------------------------------------------------------------------------
Running time
 - Total :   387.13 ms
 -  Mean :     3.87 ms
 -   STD :     1.45 ms
```

## 2. Get MACs
- [MACs](https://en.wikipedia.org/wiki/Multiply–accumulate_operation) is Multiply-accumulate operation which represents computational expense.
- Approximately 1 MACs = 0.5 * FLOPs
- Kindle is using [ptflops](https://github.com/sovrasov/flops-counter.pytorch) for computing MACs.

```python
mac = profiler.get_macs(verbose=True)
print(f"{mac:,0d} MACs")
```

```shell
Model(
  0.138 M, 100.000% Params, 0.002 GMac, 100.000% MACs, 
  (model): Sequential(
    0.138 M, 100.000% Params, 0.002 GMac, 100.000% MACs, 
    (0): Conv(
      0.001 M, 0.447% Params, 0.001 GMac, 39.517% MACs, 
      (conv): Conv2d(0.001 M, 0.435% Params, 0.001 GMac, 37.997% MACs, 3, 8, kernel_size=(5, 5), stride=(1, 1), padding=[2], bias=False)
      (batch_norm): BatchNorm2d(0.0 M, 0.012% Params, 0.0 GMac, 1.013% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.507% MACs, negative_slope=0.01)
    )
    (1): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.507% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(0.003 M, 2.321% Params, 0.001 GMac, 50.663% MACs, 8, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    (3): BatchNorm2d(0.0 M, 0.023% Params, 0.0 GMac, 0.507% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.253% MACs, inplace=True)
    (5): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.253% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, start_dim=1, end_dim=-1)
    (7): Linear(
      0.123 M, 89.220% Params, 0.0 GMac, 7.614% MACs, 
      (linear): Linear(0.123 M, 89.220% Params, 0.0 GMac, 7.607% MACs, in_features=1024, out_features=120, bias=True)
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, )
    )
    (8): Linear(
      0.01 M, 7.373% Params, 0.0 GMac, 0.634% MACs, 
      (linear): Linear(0.01 M, 7.373% Params, 0.0 GMac, 0.629% MACs, in_features=120, out_features=84, bias=True)
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, )
    )
    (9): Linear(
      0.001 M, 0.617% Params, 0.0 GMac, 0.053% MACs, 
      (linear): Linear(0.001 M, 0.617% Params, 0.0 GMac, 0.053% MACs, in_features=84, out_features=10, bias=True)
      (activation): Identity(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
  )
)
1,616,970 MACs
```

## 3. Test Time Augmentation
- Kindle model supports TTA with simple usage.
- Reference document: [https://limjk.ai/kindle/api/kindle.model/#kindle.model.Model.forward](https://limjk.ai/kindle/api/kindle.model/#kindle.model.Model.forward)

### TTA with explicit functions
```python
import torch
import torch.nn.functional as F
import numpy as np

from kindle import Model


@torch.no_grad()
def aug_flip_lr(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])

@torch.no_grad()
def aug_flip_ud(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])

@torch.no_grad()
def aug_random_scale(x: torch.Tensor) -> torch.Tensor:
    ratio = np.random.uniform(0.8, 1.2)
    xr = F.interpolate(x, scale_factor=ratio)

    if xr.shape[2:] != x.shape[2:]:
        h1, w1 = x.shape[2:]
        h2, w2 = xr.shape[2:]

        xr = F.pad(xr, [0, w1 - w2, 0, h1 - h2], value=0)

    return xr

aug_funcs = [aug_flip_lr, aug_flip_ud, aug_random_scale]
model = Model("model.yaml")

model_in = torch.rand(1, 3, 32, 32)
model_out = model(model_in, augment_func=aug_funcs)
```

- `model_out` in the above code contains **4** elements which are outputs of the model with each 3 augmentations and original `model_in`.


### TTA with repeating one augmentation function
- If a single augmentation function is given to `augment_func`, TTA will consider as random augmentation and run the network repeating `n_augment` times. 

```python
model_out = model(model_in, augment_func=aug_random_scale, n_augment = 5)
```

- `model_out` in the above code contains **6** elements which are outputs of the model with 5 `aug_random_scale` applied and original `model_in`.
