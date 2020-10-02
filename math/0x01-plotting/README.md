# 0x01. Plotting

## Learning Objectives

- Practice plots
- Scatter plot, line graph, bar graph, histogram
- What is matplotlib?
- How to plot data with matplotlib
- How to label a plot
- How to scale an axis
- How to plot multiple sets of data at the same time

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15) and `matplotlib` (version 3.0)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module

## Installing Matplotlib 3.0

```
pip install --user matplotlib==3.0
pip install --user Pillow
sudo apt-get install python3-tk
```
To check that it has been successfully downloaded, use `pip list`

## Configure X11 Forwarding

Update your `Vagrantfile` to include the following:
```
Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end
```
If you are running `vagrant` on a Mac, you will have to install `XQuartz` and restart your computer.

If you are running `vagrant` on a Windows computer, you may have to follow these instructions.

Once complete, you should simply be able to vagrant ssh to log into your VM and then any GUI application should forward to your local machine.

Hint for `emacs` users: you will have to use `emacs -nw` to prevent it from launching its GUI.

## Tasks

### [0. Line Graph](./0-line.py)

---

### [1. Scatter](./1-scatter.py)

---

### [2. Change of scale](./2-change_scale.py)

---

### [3. Two is better than one](./3-two.py)

---

### [4. Frequency](./4-frequency.py)

---

### [5. All in One](./5-all_in_one.py)

---

### [6. Stacking Bars](./6-bars.py)

---

### [7. Gradient](./100-gradient.py)

---

### [8. PCA](./101-pca.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)