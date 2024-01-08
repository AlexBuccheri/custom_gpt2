# Custom GPT2 Implementation

Learning project, following the Youtube video of
[Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s), where he implements GPT2 from
scratch, using Pytorch.

## Installation

* Set up a Conda environment that contains pytorch (not documented here).

* Modify `CONDA_ENV_NAME` in [makefile.common](makefile.common) to equal the name of the Conda environment.

* Run `make install-dev` in the project root.

### Testing

* Run pytest for the local conda env by typing `pytest` in the project root.

* Run `tox` for python versions 3.10 - 3.12 by typing `make test` in the project root.
