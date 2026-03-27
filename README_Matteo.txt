### Commands used

alias agpu="srun -c 8 --mem=128G --account=AIFAC_5C0_154 --partition=boost_usr_prod --gpus-per-node=4 -t 06:00:00 --pty bash -i"

# Interactive session

uv run --offline train --config ./config_forecasting_ERA5_CERRA.yml


### Notes
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

## CERRA variables

https://confluence.ecmwf.int/display/CKB/CERRA-Land+surface+reanalysis%3A+Data+User+Guide
*source_exclude : ['skt','tp']
*target_exclude : ['tp']


## Warning
ERA5 tp GRIB code: 228
CERRA tp GRIB code: 228228
