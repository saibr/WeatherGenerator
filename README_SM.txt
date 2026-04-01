#####################
### Commands used ###
#####################

alias agpu="srun -c 8 --mem=128G --account=AIFAC_5C0_154 --partition=boost_usr_prod --gpus-per-node=4 -t 06:00:00 --pty bash -i"

# Interactive session

uv run --offline train --config ./config_forecasting_ERA5_CERRA.yml

##################
### Evaluation ###
##################

## Training Plots
uv run --offline plot_train --from_yaml ./config/evaluate/train_plot_config.yml --output_dir ./plots/trials/
other options:
 --channels ch1 ch2 ... (default=["avg"]) 
 --streams (default=["ERA5"])
 --forecast-steps (default=[0, 1])
 --metrics (default=["mse"])


## Inference 

# agpu for few samples

# slurm job for entire test set

## Metrics

##########################
### WG-Private Changes ###
##########################

- ./hpc/leonardo_aifac/config/paths.yml
        - add "/leonardo_work/DestE_340_26/ai-ml/datasets" to data_paths

- ./hpc/leonardo_aifac/config/weathergen_slurm
    set:
        - #SBATCH --qos=<boost_qos_lprod>
        - #SBATCH --time=4-00:00:00 or 96:00:00


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
