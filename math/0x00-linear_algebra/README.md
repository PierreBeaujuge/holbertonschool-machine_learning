# 0x00. Linear Algebra

## Learning Objectives

- Manipulate vectors, matrices
- What is a transpose?
- What is the shape of a matrix?
- What is an axis?
- What is a slice?
- How do you slice a vector/matrix?
- What are element-wise operations?
- How do you concatenate vectors/matrices?
- What is the dot product?
- What is matrix multiplication?
- What is Numpy?
- What is parallelization and why is it important?
- What is broadcasting?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15)
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

## Installing Ubuntu 16.04 and Python 3.5

Follow the instructions listed in `Using Vagrant on your personal computer`, with the caveat that you should be using `ubuntu/xenial64` instead of `ubuntu/trusty64`.

Python 3.5 comes pre-installed on Ubuntu 16.04. How convenient! You can confirm this with `python3 -V`

## Installing pip 19.1

```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
```
To check that pip has been successfully downloaded, use `pip -V`. Your output should look like:
```
$ pip -V
pip 19.1.1 from /usr/local/lib/python3.5/dist-packages/pip (python 3.5)
```

## Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5

```
$ pip install --user numpy==1.15
$ pip install --user scipy==1.3
$ pip install --user pycodestyle==2.5
```
To check that all have been successfully downloaded, use `pip list`

## Tasks

### [0. Slice Me Up](./0-slice_me_up.py)

---

### [1. Trim Me Down](./1-trim_me_down.py)

---

### [2. Size Me Please](./2-size_me_please.py)

---

### [3. Flip Me Over](./3-flip_me_over.py)

---

### [4. Line Up](./4-line_up.py)

---

### [5. Across The Planes](./5-across_the_planes.py)

---

### [6. Howdy Partner](./6-howdy_partner.py)

---

### [7. Gettin’ Cozy](./7-gettin_cozy.py)

---

### [8. Ridin’ Bareback](./8-ridin_bareback.py)

---

### [9. Let The Butcher Slice It](./9-let_the_butcher_slice_it.py)

---

### [10. I’ll Use My Scale](./10-ill_use_my_scale.py)

---

### [11. The Western Exchange](./11-the_western_exchange.py)

---

### [12. Bracing The Elements](./12-bracin_the_elements.py)

---

### [13. Cat's Got Your Tongue](./13-cats_got_your_tongue.py)

---

### [14. Saddle Up](./14-saddle_up.py)

---

### [15. Slice Like A Ninja](./100-slice_like_a_ninja.py)

---

### [16. The Whole Barn](./101-the_whole_barn.py)

---

### [17. Squashed Like Sardines](./102-squashed_like_sardines.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)