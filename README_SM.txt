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
uv run --offline inference --from-run-id <RUN_ID>\
 --config ./config/inference/<config_file> \

** Slurm job ** 

../WeatherGenerator-private/hpc/launch-slurm.py \
  --stage inference \
  --nodes 1 (or 2 -- but seems to be forced to 1 for inference)
  --from-run-id <RUN_ID> \
  --config ./config/inference/inference_era_o96_config.yml \
  --register

### Evaluation ###

** Training Plots **
uv run --offline plot_train --from_yaml ./config/evaluate/train_plot_config.yml \
 --output_dir ./plots/MF_KNMI/ERA_CERRA_experiments
 --channels ch1 ch2 ... (default=["avg"]) 
 --streams ERA5 (default=["ERA5"])
 --forecast-steps 0 1 2 (default=[0, 1])
 --metrics mse (default=["mse"])

** MlFlow **
for mlflow we modified the mlflow_upload.py by overriding "weathergen.step" which is not included in the metrics dict

run ../WeatherGenerator-private/hpc/upload_experiment.py --run-id <RUN_ID> \
 --experiment-location /leonardo_work/AIFAC_5C0_154/weathergen/shared_work \
 --model no

** Compare runs **
src/weathergen/utils/compare_run_configs.py --config config/my_runs.yml

*** FastEval ***

** IS **
uv run --offline evaluate --config config/evaluate/pretrain_era5_eval_config.yml

** Slurm job ** 
you need the links, if you don't have run: ./scripts/actions.sh create-links
be aware to set properly the run_id in the config file, and wait that the job is running before making midification to it

sbatch evaluation_slurm.sh config/evaluate/{EVAL_CONFIG}.yml

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

r_: should be relative humidity
10_si: should be speed intensity
10wdir: wind direction

https://confluence.ecmwf.int/display/CKB/CERRA-Land+surface+reanalysis%3A+Data+User+Guide
*source_exclude : ['skt','tp']
*target_exclude : ['tp']


## Warning
ERA5 tp GRIB code: 228
CERRA tp GRIB code: 228228
