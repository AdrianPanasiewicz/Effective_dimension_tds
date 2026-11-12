# Structure of the files in `results`

This document describes the files stored in the `results` directory. They are produced by the depth sweep experiment and by the extension notebook that adds further circuit depths to existing results. For every circuit depth, a variational quantum model (`GQNN`) is trained and its local effective dimension (LED), global effective dimension (GED) and test loss are recorded. The experiment settings are described in the article series and in the notebooks, so they are not repeated here.

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
3. For each batch, an existing extension file is loaded and training is skipped. Otherwise each depth uses its checkpoint `trained_depth_<depth>_<seed>.pt` if it exists. If not, an Optuna study is run and the model with the lowest final validation loss is saved as a checkpoint.
4. LED, GED and test losses are computed for the batch with the same settings as the original sweep and saved as `depth_<first>_<last>_extension_<seed>.pkl`.
5. If no backup with that name exists, the main result file is copied to `...pkl.before_depth_<first>_<last>.bak`.
6. The batch is merged into the main result file, which is overwritten. All depth entries are sorted in ascending order.

Every file is written to a temporary `.tmp` file first and then moved into place. Extension files and backups are kept after the merge. The notebook is rerun with different `SEEDS` and `NEW_DEPTHS` values, so a seed can have several extension files and several backups, one for each batch.

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
| `means` | `list[float]` | Average effective dimension of depth `depths[i]` in entry `i` |
| `standard_errors` | `list[float]` | Standard error of that average in entry `i` |
| `samples` | `dict[int, list[float]]` | Raw estimates behind `means`. The dictionary key is the depth value itself and not a position, so `samples[5]` belongs to depth 5. Each value is a list with one estimate per repetition of the Monte Carlo estimation, so `samples[depth][r]` is the estimate of repetition `r`. |
| `theoretical_dataset_size` | `int` | Dataset size used in the effective dimension formula |

The values are normalized effective dimensions. The `test_losses` dictionary maps each depth to the test MSE of the best model for that depth.

### Depths contained in each file role

| File | Depths contained |
|---|---|
| Main result file | All depths merged into this seed so far |
| Extension file | Only the batch computed in that run |
| Backup file | The main file as it was before the batch in its name was merged |

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
| `state_dict["weights"]` | Trainable circuit parameters, with `n_layers` and `n_qubits` as the leading dimensions |

The configuration belongs to the Optuna trial with the lowest final validation loss at that depth. Before the extension notebook reuses a checkpoint, it checks that `n_layers` equals the depth and that `n_qubits` equals its own qubit setting. A model is rebuilt by constructing `GQNN` from `model_config` and calling `net.qlayers.load_state_dict(state_dict)`.

## Loading

```python
import pickle
from pathlib import Path

import torch

results_dir = Path("results")
seed = 2026
depth = 1

with (results_dir / f"trained_prefix_led_ged_results_{seed}.pkl").open("rb") as f:
    results = pickle.load(f)

record = torch.load(
    results_dir / f"trained_depth_{depth}_{seed}.pt",
    map_location="cpu",
    weights_only=False,
)
```
