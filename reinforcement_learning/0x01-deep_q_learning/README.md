# 0x01. Deep Q-learning

## Description

Learning Objectives

- What is Deep Q-learning?
- What is the policy network?
- What is replay memory?
- What is the target network?
- Why must we utilize two separate networks during training?
- What is keras-rl? How do you use it?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), `gym` (version 0.17.2), keras (version 2.2.5), and keras-rl (version 0.4.2)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- All your files must be executable
- Your code should use the minimum number of operations

## Installing Keras-RL

```
pip install --user keras-rl
```

## Dependencies (that should already be installed)

```
pip install --user keras==2.2.5
pip install --user Pillow
pip install --user h5py
```

## Tasks

### [0. Breakout](./train.py)

Write a python script `train.py` that utilizes `keras`, `keras-rl`, and `gym` to train an agent that can play Atari’s Breakout:

- Your script should utilize `keras-rl`‘s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`
- Your script should save the final policy network as `policy.h5`

Write a python script `play.py` that can display a game played by the agent trained by `train.py`:

- Your script should load the policy network saved in `policy.h5`
- Your agent should use the `GreedyQPolicy`

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)