Dependencies are Python 3, a recent PyTorch, numpy/scipy, tqdm, and gurobipy (along with a Gurobi license).

Code is in the module `regretnet`. To train and test auction networks, use the scripts `train.py` and `test.py`; `sample_scripts.sh` gives some examples of how to invoke these.

Pretrained models are in the `model` directory.

Certification code is in `regretnet.mipcertify`; the main experiment file is in `experiments/updated_experiment.py`.
It outputs a directory containing some diagnostic plots and a pickled list of dictionaries summarizing results for
every random point certified. It should be run using `python -m regretnet.mipcertify.experiments.updated_experiment`. Parameters
to run experiments are manually edited in the file.
