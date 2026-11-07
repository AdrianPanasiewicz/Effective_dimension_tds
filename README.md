# Effective_dimension_tds

Source code for the two-part article series on effective dimension as a measure of the learning capacity of quantum neural networks (QNNs). The project trains a QNN for Mackey-Glass time-series forecasting and analyses it with the global effective dimension (GED) and the local effective dimension (LED). Part 1 of the series covers the theory. Part 2 covers the implementation in this repository.

- **GED** averages over the entire parameter space and supports architecture-level comparisons.
- **LED** averages over an epsilon-ball around the trained weights and characterizes the trained model.

## How to run

Open `tds_experiment.ipynb` and execute the cells sequentially. The hyperparameter-search cells are computationally expensive and may be skipped, because effective hyperparameters are already provided to `BasicTrainer`.

```text
data generation
    → input-output pair creation
    → optional hyperparameter search
    → model training
    → forecasting
    → GED / LED analysis on the test set
```

The Mackey-Glass series is split chronologically into training, validation, and test sets. Parameters are optimized on the training set and hyperparameters are selected on the validation set (20 trials per circuit depth). GED, LED, and test loss are computed on the held-out test set and averaged over several seeds.

## Structure

- `core/`: the model (`core/model`), effective-dimension measures (`core/measures`), trainers (`core/trainers`), and datasets (`core/datasets`).
- `tests/`: unit tests for the effective-dimension components.
- `results/` and `charts/`: figures and experiment results.
- `tds_experiment.ipynb`: end-to-end experiment pipeline.
- `extend_depths.ipynb` and `plot_results_v2_3_withx.ipynb`: additional circuit depths and result plotting.


## Effective dimension implementation

### Classical Fisher information matrix

`core/measures/cfim.py` provides:

```python
compute_cfim_for_input(net, x, min_probability=min_probability)
```

For input $\boldsymbol{x}$ and the current parameters $\boldsymbol{\theta}$, each computational-basis outcome $y$ has probability:

$$
    p(y \mid \boldsymbol{x}; \boldsymbol{\theta}) = \left|\braket{y \mid {\psi(\boldsymbol{x}, \boldsymbol{\theta})}}\right|^2.
$$

The classical Fisher information matrix (CFIM) is computed from the gradients of these probabilities with PyTorch autograd:

$$
    \tilde{\boldsymbol{F}}_{(i,j)}(\boldsymbol{\theta})
    :=
    \frac{1}{\ell}
    \sum_{m=1}^\ell
    \left(\frac{\partial}{\partial\theta_i} \log p(\boldsymbol{x}_m,{y}_m;\boldsymbol{\theta}) \right)
    \left(
    \frac{\partial}{\partial\theta_j} \log p(\boldsymbol{x}_m,{y}_m;\boldsymbol{\theta})
    \right).
$$

Outcomes with probability below `min_probability` are ignored to prevent division by zero.


### Parameter samplers

`core/measures/samplers.py` defines samplers that implement `parameter_space.sample()`. The returned tensor has the shape of `net.qlayers.weights`.

- `UniformParameterSpace` samples every parameter independently from $\mathrm{Uniform}(\mathrm{low}, \mathrm{high})$. GED uses $\mathrm{Uniform}(0, 2\pi)$, because gate parameters are rotation angles.
- `EpsilonBallParameterSpace` samples uniformly from the ball $\|\boldsymbol{\theta} - \boldsymbol{\theta}^{\*} \|_2 \leq \epsilon$ around a centre $\boldsymbol{\theta}^{\*}$. It draws a unit direction $\boldsymbol{u}$ by normalizing a standard normal vector and a radius $r$:

$$
r = \epsilon q^{1/d},
\qquad
q \sim \mathcal{U}(0, 1),
\qquad
\boldsymbol{\theta} = \boldsymbol{\theta}^* + r \boldsymbol{u},
$$

where $d$ is the number of parameters. The $1/d$ exponent ensures uniform density over the volume of the ball. Without it, points concentrate near the centre.


### Monte Carlo empirical Fisher estimation

`core/measures/monte_carlo.py` provides:

```python
sample_empirical_fishers(
    net,
    parameter_space,
    inputs,
    n_theta=100,
    min_probability=1e-12,
)
```

For every sampled $\theta_k$, the CFIM is averaged over the inputs:

$$
\tilde{\boldsymbol{F}}(\boldsymbol{\theta}_i) = \frac{1}{N_x} \sum_{j=1}^{N_x} \boldsymbol{F}(\boldsymbol{\theta}_i, \boldsymbol{x}_j).
$$

The result has shape `(n_theta, d, d)` and the original model weights are restored afterwards. Runtime grows with the product of `n_theta` and the number of inputs.

### Effective dimension

The empirical Fisher matrices are normalized to an average trace of $d$, following Abbas et al.:

$$
\hat{\boldsymbol{F}}(\boldsymbol{\theta}_i) = \frac{k \cdot \tilde{\boldsymbol{F}}(\boldsymbol{\theta}_i)}{\mathrm{Tr}\left(\frac{1}{N_\theta} \sum_{j=1}^{N_\theta} \tilde{\boldsymbol{F}}(\boldsymbol{\theta}_j)\right)}.
$$

For a dataset size $n$, the resolution scale is 

$$
\kappa_{n,\gamma} = \frac{\gamma n}{2\pi \log n}.
$$ 

For each parameter sample $\boldsymbol{\theta}_j$, we compute the log-volume contribution:

$$
r_j=\frac{1}{2}\log\det\left[ \boldsymbol{I} + \kappa_{n,\gamma} \hat{\boldsymbol{F}}(\boldsymbol{\theta}_j) \right].
$$

Finally, our Monte Carlo estimate of the effective dimension averages these contributions over all  parameter samples:

$$
    d_{n,\gamma} = \frac{2}{\log(\kappa_{n,\gamma})} \log \left[ \frac{1}{N_\theta} \sum_{j=1}^{N_\theta} \exp(r_j) \right].
$$

The computation runs in log-space with `torch.linalg.slogdet` and `torch.logsumexp` to avoid overflow. The estimator is:

```python
estimate_effective_dimension(
    net,
    inputs,
    parameter_space,
    normalize=False,
    n_theta=100,
    min_probability=1e-12,
    sample_sizes=None,
)
```

- `parameter_space`: a `UniformParameterSpace` gives GED. An `EpsilonBallParameterSpace` centred at the trained weights gives LED.
- `sample_sizes`: the values of $n$ to evaluate. They must satisfy $n > 1$ and $\kappa_{n,\gamma} > 1$, and typically start from 100. The default is the number of inputs.
- `normalize`: if `True`, the function returns $d_{n,\gamma} / d$.

## Tests

The `tests/` directory verifies the CFIM against known analytic circuits, the shape, bounds, and reproducibility of the samplers, the Monte Carlo averaging and weight restoration, and analytic effective-dimension values for GED and LED.

```bash
python -m unittest discover -v
```

## References

1. A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, "The power of quantum neural networks," Nature Computational Science, vol. 1, no. 6, pp. 403-409, 2021. DOI: 10.1038/s43588-021-00084-1.
2. A. Abbas, D. Sutter, A. Figalli, and S. Woerner, "Effective dimension of machine learning models," arXiv:2112.04807, 2021.

To cite this code: A. Panasiewicz and J. L. Cybulski, Effective Dimension TDS: Source Code for the Experiments, GitHub repository, 2026.

Released under the MIT license.
