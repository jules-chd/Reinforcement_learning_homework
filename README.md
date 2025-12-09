# PA230 - Tournament in Reinforcement Learning
This repository contains the base for the PA230 tournament in reinforcement learning. It provides an example of the interface you should implement for submitting your agent to the tournament, a script we use to evaluate your agent, and a few snippets of code to get you started.

## Rules
The rules of the tournament can be found in [rules.pdf](rules.pdf).

## Submission
In the folder `example_submission` you can find an example of how to submit your agent. Namely, you should create a folder `agent` containing files `interface.py` and `weights.pth`, and upload it into the IS file vault `Tournament/track_name`.
The file `weights.pth` should contain the checkpoint of your trained agent, while `interface.py` should contain a class `Policy` implementing `load` and `act` methods, and a class `EnvironmentWrapper` implementing `reset` and `step` methods.


Our evaluation will be similar to the one in `eval.py`. You can check the compatibility by running `eval.py` on your submission. The following command should work from the root of this repository:
```bash
python eval.py --track 0 --load_path example_submission/Tournament/track1_dqn/agent --eps 10
```

## Example training script
In `train_example.py` you can find a simple training script that trains a Q-network to approximate the value of the uniform-random policy.
It demonstrates how to use environment wrappers, how to define and train a simple neural network in PyTorch, and how to save the trained model.
An example of a custom environment wrapper is in the script `eval.py`.

## Installation
You can find the installation instructions in [installation.md](installation.md).


## Dependencies

Please make sure your script does not import anything outside of the standard library, PyTorch, and the packages listed in `requirements.txt`.

## Virtual environment on nymfe01

You can test that your code will evaluate correctly by using our preconfigured
venv on the server `nymfe01`. 

To use it you need to be connected to the MU network, either via VPN, or just
ssh to aisa before running these commands:

```bash
ssh nymfe01 # connect
source /var/tmp/PA230/pa230/bin/activate # activate venv
python eval.py # run whatever you want to submit
```

