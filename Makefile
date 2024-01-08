# -------------------------------------------------------------------------
# Makefile for ML custom GPT project
# Note, install assumes that the python or conda environment already exists
# TODO. Add a separate command to build the env
# TODO Update and generalise clean-doc command
# -------------------------------------------------------------------------
include makefile.common

# Ensures all commands for a given target are run in the same shell
# Necessary for running installations in the specified venv
.ONESHELL:

.PHONY: install install-dev clean

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Production installation
install:
ifeq ($(ENV_TYPE), conda)
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	conda install --update-deps --yes pip setuptools wheel
else
	. $(VENV_PATH)/bin/activate
	python -m pip install --upgrade pip setuptools wheel
endif
	python -m pip install .

# Development installation
# Install dependencies specified in .dev of pyproject.toml
install-dev: pyproject.toml
ifeq ($(ENV_TYPE), conda)
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
#	conda install --update-deps --yes pip setuptools wheel
else
	. $(VENV_PATH)/bin/activate
	python -m pip install --upgrade pip setuptools wheel
endif
	python -m pip install -e ".[dev]"

# Build
build:
	python -m pip install build
	python -m build

doc:
	cd docs && \
	sphinx-apidoc -o source ../src && \
	make html

# Remove build folders
clean:
	@if [ -d ".dist" ]; then rm -r .dist; fi
	@if [ -d "src/*.egg-info" ]; then rm -r src/*.egg-info; fi

# Clean documentation
#clean-doc:
#	@if [ -f "docs/source/modules.rst" ]; then rm docs/source/modules.rst; fi
#	@if [ -f "docs/source/octopus_workflows.rst" ]; then rm docs/source/octopus_workflows.rst; fi
#	cd docs && make clean

# Apply formatting
format:
	isort $(SOURCE_DIR) tests/
	black $(SOURCE_DIR) tests/
	ruff $(SOURCE_DIR) tests/

# Check formatting
check-format:
	ruff $(SOURCE_DIR) --output-format=github
	isort --check $(SOURCE_DIR)
	black --check $(SOURCE_DIR)

# Run unit tests. Note, more pytest options are inherited from pyproject
# and tox options are specified in tox.ini
test:
	tox

# Run codecov
cov:
	pytest --cov=src/ --cov-report term --cov-fail-under=-1
