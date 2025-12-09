# Installation
We have supplied the dependencies for Homework 1 in the requirements.txt file. You have a few options when installing the required packages.

The python bindings for gym require a package called swig:

	1) sudo apt-get install swig

# Installing python requirements with conda:

The installation instructions for conda and miniconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).  After installing conda and calling `conda init`, you can create an environment and install the dependencies with the following three commands:

	1) conda create -n pa230 python=3.10
	3) conda activate pa230
	4) pip install -r requirements.txt

Note that you will need to activate this environment anytime you want to run the code.

# Installing via venv:

You can create a virtual python environment and install the dependencies there:
First install virtualenv if you don't have it yet, then create a virtual
environment called pa230 and activate it.

	1) pip install virtualenv
	2) python -m venv pa230
	3) source pa230/bin/activate

You can check that the environment was activated by calling

	which python

Then just install the requirements like in the previous step

	pip install -r requirements.txt


# System wide installation:
The last option is to just install all the dependencies system-wide without creating a virtual environment.
Note that this way you may get some dependency conflicts.

	1) pip install -r requirements.txt
			

# Torch:

To implement the neural network approximators for the value function (and
policy in the policy gradient homework), you will need to install PyTorch.

For our needs, it is perfectly sufficient to just rely on the CPU version of
pytorch, which you can download like so:

```
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

