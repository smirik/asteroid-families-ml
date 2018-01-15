* master: [![Build Status](https://travis-ci.org/4xxi/asteroid-families-ml.svg?branch=feature%2Ftravis)](https://travis-ci.org/4xxi/asteroid-families-ml)
* develop: [![Build Status](https://travis-ci.org/4xxi/asteroid-families-ml.svg?branch=develop)](https://travis-ci.org/4xxi/asteroid-families-ml)

# Asteroid families ML

## Abstract

The program uses machine learning (ML) techniques to identify the objects
as members of a concrete asteroid familily. It uses [AstDyS](http://hamilton.dm.unipi.it/astdys/)
catalog as a source of asteroid synthetic proper element database to create feature dataset,
and a source of known families to provide the target vector. The program allows to apply the
chosen ML method and get simple quality scores for it using [Scikit-learn](http://scikit-learn.org/stable/index.html).

The program is available for using in `python3` version.

The contained files are:
* config_sample.py - sample file of configuration data
* config.py        - the actual configuration data for running the program
* methods.py       - defines allowed methods, scores and metrics
* main.py          - the main executable file
* sourcelib.py     - finds and downloads needed databases and creates feature-target dataset
* processlib.py    - contains procedures for training and testing the method and getting results
* resultlib.py     - specifies the path for outout data
* source/          - source data directory
* result/          - output data directory

## Installation

At present, the instalation is available via `git clone` (you must have `git` installed previously):

`git clone https://github.com/4xxi/asteroid-families-ml.git`

Also, you must get `scikit-learn` and `pandas` installed.

## Usage

To use the program, you should set the configuration data in the `config.py` file (see `config_sample.py` for details).
In this file you can choose the method (the methods), a set of variable parameters for it and choose the family.

Then do: `python3 main.py`

If no source data exists in the source directory (as for the first time running program), the program will suggest
to download it (it can take several minutes to get the database and requires stable Internet connection).

As the source data gets available, the program will create internal dataset and run the method on a chosen set
(or sets) of possible parameters for each chosen family.

The resulting data will appear in the result directory in `.csv` format and will contain statistics for each
specified method configuration over all specified families.

The statistic datum includes: numbers of true positives (TP), true negatives (TN), false positives (FP),
false negatives (FN), and the scores of `accuracy`, `precision`, and `recall`.

