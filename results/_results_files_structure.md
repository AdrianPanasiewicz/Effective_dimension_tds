# Structure of the files in ``

This document describes the result files produced by the depth sweep on the Mackey-Glass (tau = 30) dataset and by the extension notebook that adds further depths to existing results. A variational quantum model (`GQNN`, 4 qubits) is trained at each circuit depth. Its local effective dimension (LED), global effective dimension (GED) and test loss are then recorded.

## Overview

| File | Created by | Content |
|---|---|---|
| `trained_depth_<depth>_<seed>.pt` | Original sweep and extension notebook | Best configuration and trained weights for one depth and seed |
| `trained_prefix_led_ged_results_<seed>.pkl` | Original sweep, then rewritten by the extension notebook | LED, GED and test loss for all depths of one seed |
| `depth_<first>_<last>_extension_<seed>.pkl` | Extension notebook | Metrics of one batch of newly computed depths, before merging |
| `trained_prefix_led_ged_results_<seed>.pkl.before_depth_<first>_<last>.bak` | Extension notebook | Copy of the main result file taken before that batch was merged |

## How the extension notebook produces these files

The notebook runs the following steps for every seed in `SEEDS`.

1. It loads `trained_prefix_led_ged_results_<seed>.pkl` and checks that the depths of `led_result`, `ged_result` and `test_losses` agree.
2. It compares `NEW_DEPTHS` with the depths already present. Each run of consecutive missing depths forms one batch, named by its first and last depth.
3. For each batch, an existing extension file is loaded and training is skipped. Otherwise each depth uses its checkpoint `trained_depth_<depth>_<seed>.pt` if it exists. If not, an Optuna study with 20 trials is run and the model with the lowest final validation loss is saved as a checkpoint.
4. LED, GED and test losses are computed for the batch with the same settings as the original sweep (`n_theta=10`, `repetitions=5`, `theoretical_dataset_size=1000`, `normalize=True`, `epsilon=0.1` for LED) and saved as `depth_<first>_<last>_extension_<seed>.pkl`.
5. If no backup with that name exists, the main result file is copied to `...pkl.before_depth_<first>_<last>.bak`.
6. The batch is merged into the main result file, which is overwritten. All depth entries are sorted in ascending order.

Every file is written to a temporary `.tmp` file first and then moved into place. The notebook is rerun with different `SEEDS` and `NEW_DEPTHS` values, so the file names show the values used in each run. A name such as `depth_22_22_extension_2029.pkl` means that depth 22 was the only depth in its batch for seed 2029.

## Result file layout

The main result file, the extension file and the backup share one layout. The top level is a `dict` with three keys.

```
{
    "led_result":  LEDDepthSweepResult,
    "ged_result":  GEDDepthSweepResult,
    "test_losses": {depth: float, ...},
}
```

`led_result` and `ged_result` are custom objects (`tests.led_depth_sweep.LEDDepthSweepResult` and `tests.ged_depth_sweep.GEDDepthSweepResult`) with five attributes.

| Attribute | Type | Meaning |
|---|---|---|
| `depths` | `list[int]` | Evaluated circuit depths in ascending order |
| `means` | `list[float]` | Mean effective dimension per depth, aligned with `depths` |
| `standard_errors` | `list[float]` | Standard error of the mean per depth, aligned with `depths` |
| `samples` | `dict[int, list[float]]` | Raw estimates per depth, 5 repetitions each |
| `theoretical_dataset_size` | `int` | Dataset size used in the estimation (1000) |

The values are normalized effective dimensions. The `means` entry of a depth is the average of the five values in `samples[depth]`. The `test_losses` dictionary maps each depth to the test MSE of the best model for that depth.

The three file roles differ only in the depths they contain.

| File | Depths contained |
|---|---|
| Main result file | All depths merged so far for the seed (24 depths with a maximum of 30 in the 2026 file) |
| Extension file | Only the batch computed in that run (depth 22 only in `depth_22_22_extension_2029.pkl`) |
| Backup file | The main file as it was before the batch was merged (depths 1 to 10 in the 2032 file saved before depths 11 to 15 were added) |

Unpickling these files requires the project root, which contains the `core` and `tests` packages, to be importable.

## Checkpoint layout

`trained_depth_<depth>_<seed>.pt` is a `dict` saved with `torch.save`.

```
{
    "config": {
        "model_config":    {...},
        "training_config": {...},
    },
    "state_dict": {"weights": Tensor},
}
```

| Key | Content |
|---|---|
| `config["model_config"]` | `n_layers`, `n_qubits`, `quantum_device`, `interface`, `diff_method`, `fm_style`, `meas` |
| `config["training_config"]` | `number_of_training_workers`, `number_of_validating_workers`, `number_of_testing_workers`, `batch_size`, `device`, `epochs`, `optimizer`, `regularization` |
| `state_dict["weights"]` | Trainable circuit parameters with shape `(n_layers, n_qubits, 3)` and dtype `float64` |

The configuration belongs to the Optuna trial with the lowest final validation loss at that depth. For `trained_depth_1_2026.pt` the weights tensor has shape `(1, 4, 3)`. The notebook checks that `n_layers` equals the depth and that `n_qubits` equals 4 before it reuses a checkpoint. A model is rebuilt by constructing `GQNN` from `model_config` and calling `net.qlayers.load_state_dict(state_dict)`.

## Loading

```python
import pickle
import torch

with open("results/trained_prefix_led_ged_results_2026.pkl", "rb") as f:
    results = pickle.load(f)

record = torch.load("results/trained_depth_1_2026.pt", map_location="cpu", weights_only=False)
```
