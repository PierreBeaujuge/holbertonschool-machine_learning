# 0x00. Q-learning

## Description

Learning Objectives

- What is a Markov Decision Process?
- What is an environment?
- What is an agent?
- What is a state?
- What is a policy function?
- What is a value function? a state-value function? an action-value function?
- What is a discount factor?
- What is the Bellman equation?
- What is epsilon greedy?
- What is Q-learning?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), and `gym` (version 0.7)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- All your files must be executable
- Your code should use the minimum number of operations

## Installing OpenAI’s Gym

```
pip install --user gym
```

## Tasks

### [0. Load the Environment](./0-load_env.py)

Write a function `def load_frozen_lake(desc=None, map_name=None, is_slippery=False):` that loads the pre-made `FrozenLakeEnv` evnironment from OpenAI’s `gym`:
- `desc` is either `None` or a list of lists containing a custom description of the map to load for the environment
- `map_name` is either `None` or a string containing the pre-made map to load
- Note: If both `desc` and `map_name` are `None`, the environment will load a randomly generated 8x8 map
- is_slippery is a boolean to determine if the ice is slippery
- Returns: the environment

```
$ ./0-main.py
[[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'H' b'F' b'H' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'H' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'H' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'G']]
[(1.0, 0, 0.0, False)]
[[b'S' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
 [b'F' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'G']]
[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, True)]
[[b'S' b'F' b'F']
 [b'F' b'H' b'H']
 [b'F' b'F' b'G']]
[[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
$
```

---

### [1. Initialize Q-table](./1-q_init.py)

Write a function `def q_init(env):` that initializes the Q-table:
- `env` is the `FrozenLakeEnv` instance
- Returns: the Q-table as a `numpy.ndarray` of zeros

```
$ ./1-main.py
(64, 4)
(64, 4)
(9, 4)
(16, 4)
$
```

---

### [2. Epsilon Greedy](./2-epsilon_greedy.py)

Write a function `def epsilon_greedy(Q, state, epsilon):` that uses epsilon-greedy to determine the next action:
- `Q` is a `numpy.ndarray` containing the q-table
- `state` is the current state
- `epsilon` is the epsilon to use for the calculation
- You should sample `p` with `numpy.random.uniformn` to determine if your algorithm should explore or exploit
- If exploring, you should pick the next action with `numpy.random.randint` from all possible actions
- Returns: the next action index

```
$ ./2-main.py
2
0
$
```

---

### [3. Q-learning](./3-q_learning.py)

Write the function `def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs Q-learning:
- `env` is the `FrozenLakeEnv` instance
- `Q` is a `numpy.ndarray` containing the Q-table
- `episodes` is the total number of episodes to train over
- `max_steps` is the maximum number of steps per episode
- `alpha` is the learning rate
- `gamma` is the discount rate
- `epsilon` is the initial threshold for epsilon greedy
- `min_epsilon` is the minimum value that `epsilon` should decay to
- `epsilon_decay` is the decay rate for updating `epsilon` between episodes
- When the agent falls in a hole, the reward should be updated to be `-1`
- Returns: `Q, total_rewards`
  - `Q` is the updated Q-table
  - `total_rewards` is a list containing the rewards per episode

```
$ ./3-main.py
[[ 0.96059593  0.970299    0.95098488  0.96059396]
 [ 0.96059557 -0.77123208  0.0094072   0.37627228]
 [ 0.18061285 -0.1         0.          0.        ]
 [ 0.97029877  0.9801     -0.99999988  0.96059583]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.98009763  0.98009933  0.99        0.9702983 ]
 [ 0.98009922  0.98999782  1.         -0.99999952]
 [ 0.          0.          0.          0.        ]]
500 : 0.812
1000 : 0.88
1500 : 0.9
2000 : 0.9
2500 : 0.88
3000 : 0.844
3500 : 0.892
4000 : 0.896
4500 : 0.852
5000 : 0.928
$
```

---

### [4. Play](./4-play.py)

Write a function `def play(env, Q, max_steps=100):` that has the trained agent play an episode:
- `env` is the `FrozenLakeEnv` instance
- `Q` is a `numpy.ndarray` containing the Q-table
- `max_steps` is the maximum number of steps in the episode
- Each state of the board should be displayed via the console
- You should always exploit the Q-table
- Returns: the total rewards for the episode

```
$ ./4-main.py

`S`FF
FHH
FFG
  (Down)
SFF
`F`HH
FFG
  (Down)
SFF
FHH
`F`FG
  (Right)
SFF
FHH
F`F`G
  (Right)
SFF
FHH
FF`G`
1.0
$
```

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)