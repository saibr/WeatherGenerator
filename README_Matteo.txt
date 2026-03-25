alias agpu="srun -c 8 --mem=128G --account=AIFAC_5C0_154 --partition=boost_usr_prod --gpus-per-node=4 -t 06:00:00 --pty bash -i"

### Interactive session

uv run --offline train --config ./config_forecasting_ERA5_CERRA.yml