'''
This is the sample of configuration file.

All parameters presented here must be presented in the configuration file.

Methods allowed for use at present:
* KNeighborsClassifier
* GradientBoostingClassifier

Parameter descriptions:
* ALG - the chosen method

* VAR_PARAMS - the 'dict' object containing a set of parameters for the chosen method.
    Here, you can pass a single value or list of values to one or several of parameters.
    In last case, the program will run several times for each possible configuration of values.

* FIRST_FAMILY - Defines the first family (the number of it's main member object) to be processed
     The list of possible families is generated automatically from the source database.

* LAST_FAMILY - Defines the last family to be processed.
    If set to 'None', only one family specified by FIRST_FAMILY will be processed

* VERBOSE - defines how much information will appear in console output (variating from 1 to 6)
* N_SAMPLES - specifies how many times each method configuration should be run.
    The resulting scores will be calculated as averages for all those samples.
'''

from methods import *


ALG = KNeighborsClassifier
VAR_PARAMS = {
             'n_neighbors': [3, 5, 7],
             'metric': 'minkowski',
             'p': 2,
             'n_jobs': 6
             }

FIRST_FAMILY = 2
LAST_FAMILY = None

VERBOSE = 3
N_SAMPLES = 1

