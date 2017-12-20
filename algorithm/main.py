import pandas as pd

from methods import *
import processlib
import sourcelib
try:
    import config
except Exception:
    raise SystemExit('Wrong configuration file format (please check your configuration using config_sample as an example). Terminating the program.')


if config.ALG not in ALLOWED_METHODS:
    raise SystemExit('Wrong method specified in the config file (please check your configuration using config_sample as an example). Terminating the program.')

# Ensuring that the source data is available
sourcelib.check_source()
data = pd.read_csv(sourcelib.DATA_SOURCE)
sourcelib.get_possible_families(data)

# Get input parameters
for key, item in config.VAR_PARAMS.items():
    if not hasattr(item, '__getitem__') or hasattr(item, 'rstrip'):
        config.VAR_PARAMS[key] = [config.VAR_PARAMS[key]]

config_args = (config.FIRST_FAMILY,
               config.LAST_FAMILY,
               config.ALG,
               config.VAR_PARAMS)

config_kwargs = {'verbose': config.VERBOSE,
                 'n_samples': config.N_SAMPLES}

# Get results
processlib.custom_process(data, *config_args, **config_kwargs)

print('All processes finished.')

