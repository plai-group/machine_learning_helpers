# A bunch of helper functions for machine learning
An assortment of helper functions for machine learning. Heavily overfit to my coding idiosyncrasies, and meant to be used in conjunction with my [project template](https://github.com/vmasrani/ml_project_skeleton).
To have these helpers be globally available for all your projects, create a hidden directory in home (i.e. `~/.python`) and set your `PYTHONPATH` in your bashrc via:

`export PYTHONPATH=~/.python:$PYTHONPATH`

Then you can use `import ml_helpers as mlh` in all your projects. For notebooks, store your boilerplate code in `init.ipynb` and run 

`%run ~/.python/init` at the beginning of each new notebook.
