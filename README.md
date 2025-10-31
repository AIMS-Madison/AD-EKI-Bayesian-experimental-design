# AD-EKI

This repository supports the paper *"Bayesian Experimental Design for Model Discrepancy Calibration: An Auto-Differentiable Ensemble Kalman Inversion Approach"*


---

### Environment Requirements

- `jax`         0.4.20 ~ 0.4.28  
- `jaxlib`      0.4.18+cuda12.cudnn89 ~ 0.4.28+cuda12.cudnn89  
- `jax-cfd`     0.2.0  
- `flax`        0.8.2  
- `optax`       0.2.2 ~ 0.2.4  

⚠️ *Note: `optax` 0.2.4 is only compatible with `jax` 0.4.28. Please ensure version consistency.*

⚠️ *The exact versions used to generate the results are specified in detail in the code.*

Experiments in the paper were originally run on **WSL with an RTX 4090**. The code is also known to **exceed memory limits** on a 12GB **4070 Super** due to a long solving trajectory. CPU execution is possible but may be slower.

---

### Contents

- `Structural_error_ensemble_BED.ipynb`:  
  Jupyter notebook to reproduce Figs. 6(a), 7 and 8 in the paper.

- `jax_cfd_test/`:  
  Necessary components for the custom PDE solver:
  - `my_equations.py`: core PDE-solving framework  
  - `my_forcing.py`: forcing term used in the PDE  
  - `my_funcutils.py`: utility functions for the AD-based solver

- `running_demo.ipynb`:
  a demo of applying AD-EKI

---

### Citation
If you use this code or find it helpful for your research, please cite:

```bibtex
@article{YANG2025114469,
  title = {Bayesian Experimental Design for Model Discrepancy Calibration: An Auto-Differentiable Ensemble Kalman Inversion Approach},
  journal = {Journal of Computational Physics},
  pages = {114469},
  year = {2025},
  issn = {0021-9991},
  doi = {https://doi.org/10.1016/j.jcp.2025.114469},
  url = {https://www.sciencedirect.com/science/article/pii/S002199912500751X},
  author = {Huchen Yang and Xinghao Dong and Jin-Long Wu},
}
