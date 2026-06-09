#####################
### Commands used ###
#####################

### Training ###

** Interactive Session (IS)**

uv run --offline train --config ./config_forecasting_ERA5_CERRA.yml
to continue specify also --from-run-id --run-id --mini-epoch 48 otherwise continue from last_checkpoint
furthermore in the general training config set:
        - istep to the last istep before the last saved checkpoint
        - run_history as iterables of tuples of run_id and istep of last saved checkpoint
        - be aware that world size must be equal otherwise mini epoch count will be no more consistent

** Slurm job **
../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 \
 --config config/config_forecasting_era5_cerra.yml \
 --chain-jobs <int> \
 --register

### Inference ###

** IS for few samples **
uv run --offline inference --from-run-id <RUN_ID>\
 --config ./config/inference/<config_file> \
 --mini-epoch
 --time=12:00:00 \ otherwise it is just 1h

** Slurm job ** 

../WeatherGenerator-private/hpc/launch-slurm.py \
  --stage inference \
  --run-id 
  --mini-epoch
  --nodes 1 (or 2 -- but seems to be forced to 1 for inference)
  --from-run-id <RUN_ID> \
  --config ./config/inference/inference_era_o96_config.yml \
  --register or --no-register
  --time=12:00:00 \ otherwise it is just 1h

### Evaluation ###

** Training Plots **
uv run --offline plot_train --from_yaml ./config/evaluate/train_plot_config.yml \
 --output_dir ./plots/MF_KNMI/ERA_CERRA_experiments
 --channels ch1 ch2 ... (default=["avg"]) 
 --streams "ERA5" (default=["ERA5"])
 --forecast-steps 0 1 2 (default=[0, 1])
 --metrics mse (default=["mse"])
 --per-stream-y-lim 1e-3 1 (two int)
 --per-stream-x-lim 0 200000 (two int)

uv run --offline plot_train --from_yaml ./config/evaluate/train_plot_config_CERRA.yml --output_dir ./plots/MF_KNMI/ERA_CERRA_experiments/hedgedoc/headline_scores/1/CERRA_loss_plots/ --channels 2t 10si 10wdir r_850 t_850  u_850 v_850 z_500 --streams "CERRA" --forecast-steps 0 1 2 --metrics mse

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

##########################
### WG-Private Changes ###
##########################

For Leonardo:
- ./hpc/leonardo_aifac/config/paths.yml
        add:
                - "/leonardo_work/DestE_340_26/ai-ml/datasets" to data_paths

- ./hpc/leonardo_aifac/config/weathergen_slurm
        set:
                - #SBATCH --qos=<boost_qos_lprod>
                - #SBATCH --time=4-00:00:00 or 96:00:00

For jupiter, if you have account or project permission errors when launching slurm jobs:
- ./hpc/leonardo_aifac/config/paths.yml
        set:
                post_train:
                        slurm_account: "<YOUR-PROJECT>"

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
Cristian Lusanna Geoinfo
#############

You need the anemoi integration in multi_stream_data_sampler.py

We removed one check (give error):
self.check_same_grid(d1, d2) commented in anemoi/datasets/data

#############
FastEval Changes
(packages/evaluate)
#############

### Exporter ###

To export GRIB we (MF) have commented two lines in export/parsers/quaver_parser.py

252    "expver": self.expver,
253    "marsClass": "rd",

### Regions ###

Added AROME region into utils/regions.py, line 32
"arome": (37.0, 56.0, -12.0, 16.0),

###############
plot_training
###############

We (MF) produced a version of plot_training with the possibility of:
- specify colors
- merge runs

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
lsm: land sea mask
msl: mean sea level pressure
z_: 

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
