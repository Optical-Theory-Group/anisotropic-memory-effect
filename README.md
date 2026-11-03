# Anisotropy in Fourier space optical memory effect correlations

Simulation code for

N. Byrnes and M. R. Foreman, "Anisotropy in Fourier space optical memory effect correlations", [arXiv:2605.03469](https://arxiv.org/abs/2605.03469) (2026).

The paper studies the Fourier-domain speckle correlations behind the optical memory effect in disordered media. A single scattering theory shows that the correlation strength depends on differences in the axial wavevector components as well as on the transverse wavevector shift. The same anisotropy appears in the pseudo-correlations of the conjugate memory effect. This repository holds the simulations of a three-dimensional single scattering medium that test the theory.

## Contents

| File | Description |
|---|---|
| `plane_wave.py` | Simulation module. Places point scatterers at random in a slab and computes the incident plane wave and the singly scattered field on a measurement plane, in transmission or reflection. Runs on a GPU with CuPy or on a CPU with NumPy. |
| `plane_wave.ipynb` | Notebook that sets the parameters, computes angular spectra with a two-dimensional chirp z-transform and runs the three numerical experiments. |
| `LICENSE` | GNU General Public License v3.0. |

## Model

The scatterers are points placed uniformly at random in a slab. Each one is driven by the incident plane wave and radiates a scalar spherical wave $e^{ikr}/kr$. The scattered field is the sum of these waves, so the model includes single scattering only. The field is sampled on a plane just beyond the slab, in transmission or in reflection. Its angular spectrum is computed for transverse wavevector components from $-k$ to $k$.

The table lists the default values in the notebook. Lengths are in units of the wavelength.

| Parameter | Value |
|---|---|
| Wavelength | 500 nm |
| Slab size | 1000 × 1000 × 10 |
| Number of scatterers | 1000 |
| Measurement planes | 1000 × 1000, placed 5 beyond each face of the slab |
| Sampling interval | 0.5, giving 2000 × 2000 points |
| Random seed | 0 |

## Numerical experiments

Each experiment uses two incident plane waves chosen on the grid of transverse wavevectors. The notebook computes the angular spectrum of the scattered field for each wave, forms the local correlation of the two spectra in a 3 × 3 window and compares it with the theory. Each experiment saves its figure as an SVG file in the working folder.

1. **Experiment 1, tilting.** The second wave is tilted relative to the first. One transmitted spectrum is shifted by the tilt before the correlation is formed. Output `transmission_band_final.svg`.
2. **Experiment 2, tt′ loop correlations.** The axial wavevector component of the second wave is reversed, so the two waves cross the slab in opposite directions. The fields transmitted for the two waves are correlated. Output `transmission_ring_final.svg`.
3. **Experiment 3, tt loop pseudo-correlations.** The two incident waves sit symmetrically about a centre point in the transverse wavevector plane. One transmitted spectrum is inverted through the centre point before the correlation is formed. Output `pseudo_ring.svg`.

Set `calibrate = True` near the top of the notebook to plot example speckle patterns and angular spectra, which help in choosing the window sizes.

The cell after Experiment 3 contains `assert False`, which stops "Run All" at that point. The cells below it hold further visualisations, animations and a loop over many disorder realisations.

## Requirements

- Python 3.10 or later. The code was developed with Python 3.12.
- NumPy, SciPy, Matplotlib and tqdm.
- CuPy and a CUDA-capable NVIDIA GPU. The notebook as written needs the GPU, because it copies results back with `.get()`. `plane_wave.py` also runs on a CPU with `use_cupy=False`, but it imports CuPy, so CuPy must be installed.
- Jupyter. The cells below `assert False` also use ipywidgets, ipympl and ffmpeg.

Install the packages with

```bash
pip install numpy scipy matplotlib tqdm jupyter ipywidgets ipympl
pip install cupy-cuda12x    # pick the CuPy package that matches your CUDA version
```

## Usage

Clone the repository and start Jupyter from its folder, so that the notebook can import `plane_wave.py`. Then run the cells in order.

```bash
git clone https://github.com/Optical-Theory-Group/anisotropic-memory-effect.git
cd anisotropic-memory-effect
jupyter lab plane_wave.ipynb
```

`plane_wave.py` can also be used on its own. The example below computes one speckle pattern on a CPU with the default geometry of the notebook.

```python
import numpy as np
import plane_wave

# Lengths are in units of the wavelength
wavelength = 500e-9
medium = plane_wave.MediumParameters(1000, 1000, 10, 1000)                 # x, y, z extent, number of scatterers
plane = plane_wave.MeasurementPlaneParameters(1000, 1000, 10, 2000, 2000)  # x, y extent, z position, points in x and y
sp = plane_wave.SimulationParameters(wavelength, False, medium, plane, seed=0)  # False selects NumPy

x, y, z = plane_wave.get_particle_locations(sp)
k_in = np.array([0.0, 0.0, 1.0])  # direction of the incident wave
field = plane_wave.get_scattered_field(x, y, z, k_in, sp)
```

## Citation

If you use this code, please cite the paper.

```bibtex
@misc{Byrnes2026Anisotropy,
  author        = {Byrnes, Niall and Foreman, Matthew R.},
  title         = {Anisotropy in {F}ourier space optical memory effect correlations},
  year          = {2026},
  eprint        = {2605.03469},
  archivePrefix = {arXiv},
  primaryClass  = {physics.optics},
  doi           = {10.48550/arXiv.2605.03469}
}
```

## Licence

GNU General Public License v3.0. See [LICENSE](LICENSE).

## Contact

Optical Theory Group, [www.mrforeman.com](https://www.mrforeman.com).
