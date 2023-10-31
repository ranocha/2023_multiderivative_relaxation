# Multiderivative time integration methods preserving nonlinear functionals via relaxation

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{ranocha2023multiderivative,
  title={Multiderivative time integration methods preserving nonlinear
         functionals via relaxation},
  author={Ranocha, Hendrik and Sch{\"u}tz, Jochen},
  year={2023},
  doi={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2023multiderivativeRepro,
  title={Reproducibility repository for
         "{M}ultiderivative time integration methods preserving nonlinear
         functionals via relaxation"},
  author={Ranocha, Hendrik and Sch{\"u}tz, Jochen},
  year={2023},
  howpublished={\url{https://github.com/ranocha/2023_multiderivative_relaxation}},
  doi={TODO}
}
```

## Abstract

We combine the recent relaxation approach with multiderivative Runge-Kutta methods
to preserve conservation or dissipation of entropy functionals for ordinary and
partial differential equations. Relaxation methods are minor modifications of
explicit and implicit schemes, requiring only the solution of a single scalar
equation per time step in addition to the baseline scheme. We demonstrate the
robustness of the resulting methods for a range of test problems including the
3D compressible Euler equations. In particular, we point out improved error growth
rates for certain entropy-conservative problems including nonlinear dispersive
wave equations.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.9.3.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code_julia` directory of this repository and follow the instructions
described in the `README.md` file therein.
Other numerical experiments use MATLAB, based on source code contained in the
directory `code_matlab`.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)
- Jochen Schütz (Hasselt University, Belgium)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file. Some parts of the implementation are inspired by corresponding
code of [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
published also under the MIT license, see
[their license file](https://github.com/SciML/OrdinaryDiffEq.jl/blob/780c94aa8944979d9dcbfb0e34c1f2554727a471/LICENSE.md).


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
