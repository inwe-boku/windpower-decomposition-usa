[![MIT License](https://img.shields.io/github/license/inwe-boku/wind-repowering-usa.svg)](https://choosealicense.com/licenses/mit/)
[![Tests](https://github.com/inwe-boku/windpower-decomposition-usa/workflows/Tests/badge.svg)](https://github.com/inwe-boku/windpower-decomposition-usa/actions?query=workflow%3ATests)
[![DOI](https://img.shields.io/badge/DOI-10.1088/2515-7620/ace0b9-blue)](https://doi.org/10.1088/2515-7620/ace0b9)


Explaining the decline of US wind output power density
======================================================

Peter Regner, Katharina Gruber, Sebastian Wehrle, Johannes Schmidt

This repository contains code, figures and results for a [study regarding US wind power density](https://doi.org/10.1088/2515-7620/ace0b9).


Abstract
--------

US wind power generation has grown significantly over the last decades, in line with the number and average size of operating turbines. However, wind power density has declined, both measured in terms of wind power output per rotor swept area as well as per spacing area. To study this effect, we present a decomposition of US wind power generation data for the period 2001--2021 and examine how changes in input power density and system efficiency affected output power density. Here, input power density refers to the amount of wind available to turbines, system efficiency refers to the share of power in the wind flowing through rotor swept areas which is converted to electricity and output power density refers to the amount of wind power generated per rotor swept area. We show that, while power input available to turbines has increased in the period 2001--2021, system efficiency has decreased. In total, this has caused a decline in output power density in the last 10 years, explaining higher land-use requirements. The decrease in system efficiency is linked to the decrease in specific power, i.e. the ratio between the nameplate capacity of a turbine and its rotor swept area. Furthermore, we show that the wind available to turbines has increased substantially due to increases in the average hub height of turbines since 2001. However, site quality has slightly decreased in this period.


Results
-------

All final computation results are available as NetCDF files:

| File                                                                                               | Description                                                                                                  | Unit            |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------- |
| [p_in.nc](data/output/results/p_in.nc)                                                             | Power input (actual wind conditions, hub height)                                                             | GW              |
| [p_in_avgwind.nc](data/output/results/p_in_avgwind.nc)                                             | Power input (long-term average wind conditions, hub height)                                                  | GW              |
| [p_in_avgwind_refheight.nc](data/output/results/p_in_avgwind_refheight.nc)                         | Power input (long-term average wind conditions, reference height)                                            | GW              |
| [p_out_model.nc](data/output/results/p_out_model.nc)                                               | Power output (actual wind conditions, hub height)                                                            | GW              |
| [p_out_model_avgwind.nc](data/output/results/p_out_model_avgwind.nc)                               | Power output (long-term average wind conditions, hub height)                                                 | GW              |
| [p_out_model_avgwind_refheight.nc](data/output/results/p_out_model_avgwind_refheight.nc)           | Power output (long-term average wind conditions, reference height)                                           | GW              |
| [p_out_model_aging.nc](data/output/results/p_out_model_aging.nc)                                   | Power output (actual wind conditions, hub height, aging loss subtracted)                                     | GW              |
| [p_out_model_aging_avgwind.nc](data/output/results/p_out_model_aging_avgwind.nc)                   | Power output (long-term average wind conditions, hub height, aging loss subtracted)                          | GW              |
| [d_out.nc](data/output/results/d_out.nc)                                                           | Input power density (actual wind conditions, hub height)                                                     | W/m^2           |
| [d_out_avgwind.nc](data/output/results/d_out_avgwind.nc)                                           | Input power density (long-term average wind conditions, hub height)                                          | W/m^2           |
| [d_in.nc](data/output/results/d_in.nc)                                                             | Input power density (actual wind conditions, hub height)                                                     | W/m^2           |
| [d_in_avgwind.nc](data/output/results/d_in_avgwind.nc)                                             | Input power density (long-term average wind conditions, hub height)                                          | W/m^2           |
| [d_in_avgwind_refheight.nc](data/output/results/d_in_avgwind_refheight.nc)                         | Input power density (long-term average wind conditions, reference height)                                    | W/m^2           |
| [num_turbines_built.nc](data/output/results/num_turbines_built.nc)                                 | Number of operating turbines                                                                                 | dimensionless   |
| [rotor_swept_area.nc](data/output/results/rotor_swept_area.nc)                                     | Total rotor swept area of operating turbines                                                                 | m^2             |
| [rotor_swept_area_avg.nc](data/output/results/rotor_swept_area_avg.nc)                             | Average rotor swept area for operating turbines                                                              | m^2             |
| [efficiency.nc](data/output/results/efficiency.nc)                                                 | System efficiency of operating turbines, i.e. P_out/P_in (actual wind conditions)                            | dimensionless   |
| [efficiency_avgwind.nc](data/output/results/efficiency_avgwind.nc)                                 | System efficiency of operating turbines, i.e. P_out,avgwind/P_in,avgwind (long-term average wind conditions) | dimensionless   |
| [specific_power_per_year.nc](data/output/results/specific_power_per_year.nc)                       | Average specific power of operating turbines per year                                                        | W/m^2           |
| [total_capacity_kw.nc](data/output/results/total_capacity_kw.nc)                                   | Total capacity of operating turbines                                                                         | KW              |
| [capacity_factors_model.nc](data/output/results/capacity_factors_model.nc)                         | Capacity factors (simulation using power curve model, actual wind conditions)                                | %               |
| [capacity_factors_model_avgwind.nc](data/output/results/capacity_factors_model_avgwind.nc)         | Capacity factors (simulation using power curve model, long-term average wind conditions)                     | %               |
| [capacity_factors_eia.nc](data/output/results/capacity_factors_eia.nc)                             | Capacity factors (using observation data provided by EIA)                                                    | %               |
| [avg_hubheights.nc](data/output/results/avg_hubheights.nc)                                         | Average hub heights of operating turbines                                                                    | m               |


Requirements
------------

To reproduce the results

* dependencies: see [env.yml](env.yml) + standard tools like GNU Make, wget; tested only with
  GNU/Linux
* approx. 200GB of disk space, might require also >16GB RAM (depends on which version is run)
* API key for the [CDS API](https://cds.climate.copernicus.eu/api-how-to)
* API key for the [EIA API](https://www.eia.gov/developer/)


The high-level dependencies can be installed by running:

```
conda install -c conda-forge mamba
mamba install -c conda-forge python=3.8 matplotlib pytest-cov dask openpyxl pytest pip xarray netcdf4 jupyter pandas scipy flake8 dvc pre-commit pyarrow statsmodels rasterio scikit-learn
# optional dependencies:
mamba install -c conda-forge pytest-watch pdbpp black seaborn
pip install cdsapi
```

Note: at the moment we need Python 3.8 because of dvc.


How to run
----------

[DVC](dvc.org/) is used to track versions and execute all stages in the computation pipeline. It
installed using conda.

Preparation:

* Request an API key for the [CDS API](https://cds.climate.copernicus.eu/api-how-to) and store it
  `~/.cdsapirc` as [described](https://cds.climate.copernicus.eu/api-how-to).
* Request an API key for the [EIA API](https://www.eia.gov/developer/) and store it in a plain text
  file `eia-api-key` in the repository root directory.
* Create a conda environment and install dependencies:
```
conda env update -f env.yml
conda activate wind_power_decomposition_usa
```

Download data:
* `dvc repro download_turbines`: unfortunately old versions of the USWTDB are not available for
  download, i.e. this won't work automatically. A recent version of the USWTDB is available [here](https://eerscmap.usgs.gov/uswtdb/data/), [load_data.py](src/load_data.py) has to be adapted to the the new USWTDB file.
* `dvc repro download_wind_era5` - will download approx. 115GB of ERA5 wind speed data
* `dvc repro download_p_out_eia`


Run computations and generate figures:
* `dvc repro figures`


Tests
-----

There are unit tests which don't require any data in `tests/no_data_dependency` and unit tests
which assume that the computation pipeline was run already in `tests/external_data_dependency`.
Also a lint DVC stage exists and a DVC stage which runs all notebooks. See [dvc.yaml](dvc.yaml) for
details.

All DVC stages post-fixed with `_simulation` run the whole computation pipeline using synthetic
random data to test the results. This is run with the name _integrationtest_ as Github action.


Pre-commit hooks
----------------

Pre-commit hooks are not required to run computations, but they are recommended for development.
[Pre-commit](https://pre-commit.com/) is used to manage and maintain pre-commit hooks.
Useful commands include:
- `$ pre-commit install` to install the hooks
- `$ pre-commit run --all-files` to run manually on all files (required on initial run)
