#####################
### Commands used ###
#####################

### Training ###

** Interactive Session (IS)**

uv run --offline train --config ./config_forecasting_ERA5_CERRA.yml

** Slurm job **
../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --config config/config_forecasting_era5_cerra.yml --register

### Inference ###

** IS for few samples **
uv run --offline inference --from-run-id {RUN_ID} \
 --options test_config.start_date=2023-10-01T00:00 \
 test_config.end_date=2023-12-31T00:00 \
 test_config.output.num_samples=1e16 \
 test_config.samples_per_mini_epoch=1e4 \
 streams_directory=./config/inference/streams/era5_o96

** Slurm job ** 

../WeatherGenerator-private/hpc/launch-slurm.py \
  --stage inference \
  --nodes 1 (or 2 -- but seems to be forced to 1 for inference)
  --from-run-id <RUN_ID> \
  --config ./config/inference/inference_era_o96_config.yml \
  --register

### Evaluation ###

** Training Plots **
uv run --offline plot_train --from_yaml ./config/evaluate/train_plot_config.yml --output_dir ./plots/trials/
other options:
 --channels ch1 ch2 ... (default=["avg"]) 
 --streams (default=["ERA5"])
 --forecast-steps (default=[0, 1])
 --metrics (default=["mse"])

for mlflow we modified the mlflow_upload.py by overriding "weathergen.step" which is not included in the metrics dict

** Compare runs **
src/weathergen/utils/compare_run_configs.py --config config/my_runs.yml

** FastEval **
in IS
uv run --offline evaluate --config config/evaluate/pretrain_era5_eval_config.yml

** Metrics **
-  Quaver ???

##########################
### WG-Private Changes ###
##########################

- ./hpc/leonardo_aifac/config/paths.yml
        add:
                - "/leonardo_work/DestE_340_26/ai-ml/datasets" to data_paths

- ./hpc/leonardo_aifac/config/weathergen_slurm
        set:
                - #SBATCH --qos=<boost_qos_lprod>
                - #SBATCH --time=4-00:00:00 or 96:00:00

- ./hpc/mlflow_upload.py
        to visualize plots in mlflow:
                - "weathergen.step" must be overrided, not present in metrics

- ./hpc/mlflow_upload.py
        remove package version:
                - "certifi"

- ./hpc/launch_slurm.py
        remove package version:
                - "certifi"

#############
### Notes ###
#############

## ERA5 variables
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
z_: geopotential
u_,v_: wind component
q_: specific umidity
t_: temperature
r_: relative humidity
tcwv: Total column water vapour

*source_exclude : ['w_', 'skt', 'tcw', 'cp', 'tp']
w_: Vertical velocity
skt: Skin temperature
tcw: Total column water
cp: Convective precipitation
tp: Total precipitation

*target_exclude : ['w_', 'slor', 'sdor', 'tcw', 'cp', 'tp']
w_: Vertical velocity
slor: Slope of sub-gridscale orography
sdor: Standard deviation of sub-gridscale orography
tcw: Total column water
cp: Convective precipitation
tp: Total precipitation
tciwv: Total column integrated water vapour

## CERRA variables
snow related: 'al' 'rsn', 'sde', 'sf' 

https://confluence.ecmwf.int/display/CKB/CERRA-Land+surface+reanalysis%3A+Data+User+Guide
*source_exclude : ['skt','tp']
*target_exclude : ['tp']


## Warning
ERA5 tp GRIB code: 228
CERRA tp GRIB code: 228228
