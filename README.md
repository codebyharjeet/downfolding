<div align="left">
  <img src="docs/src/ducc_logo.png" height="60px"/>
</div>

# DUCC

A python package to run double unitary coupled cluster (DUCC) calculations.


[//]: # (Badges)
[//]: # [![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/ducc/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/ducc/actions?query=workflow%3ACI)
[//]: #[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Double/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Double/branch/main)


### Installation
1. Download

        git clone https://github.com/codebyharjeet/DUCC.git
        cd DUCC

2. Create conda environment

        conda create --name ducc_env "python=3.7.12"
        conda activate ducc_env

3. To do the installation in development mode

        pip install -e .

4. Try running tests to make sure it is working

        pytest test/*.py
