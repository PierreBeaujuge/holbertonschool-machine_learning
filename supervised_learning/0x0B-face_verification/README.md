# 0x0B. Face Verification

## Project upload pending (awaiting server fixes and review)

## Learning Objectives

- What is face recognition?
- What is face verification?
- What is `dlib`?
- How to use `dlib` for detecting facial landmarks
- What is an affine transformation?
- What is face alignment?
- Why would you use facial alignment?
- How to use `opencv-python` for face alignment
- What is one-shot learning?
- What is triplet loss?
- How to create custom Keras layers
- How to add losses to a custom Keras layer
- How to create a Face Verification system
- What are the ethical quandaries regarding facial recognition/verification?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), tensorflow (version 1.12), `opencv-python` (version 4.1.0.25), and `dlib` (version 19.17.0)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)

## Install dlib 19.17.0

```
sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libx11-dev libgtk-3-dev
sudo apt-get install -y python3 python3-dev python3-pip
pip install --user dlib==19.17.0
```
If the installation process does not work, you may need to increase your swap memory.

## Testing Models

- `face_verification.h5` based on `swghosh/DeepFace`
- `shape_predictor_68_face_landmarks.dat.bz2`
- `HBTN.tar.gz`
- `FVTriplets.csv`

## Tasks

### [0. Load Images](./utils.py)

---

### [1. Load CSV](./utils.py)

---

### [2. Initialize Face Align](./align.py)

---

### [3. Detect Faces](./align.py)

---

### [4. Find Landmarks](./align.py)

---

### [5. Align Faces](./align.py)

---

### [6. Save Files](./utils.py)

---

### [7. Generate Triplets](./utils.py)

---

### [8. Initialize Triplet Loss](./triplet_loss.py)

---

### [9. Calculate Triplet Loss](./triplet_loss.py)

---

### [10. Call Triplet Loss](./triplet_loss.py)

---

### [11. Initialize Train Model](./train_model.py)

---

### [12. Train](./train_model.py)

---

### [13. Save](./train_model.py)

---

### [14. Calculate Metrics](./train_model.py)

---

### [15. Best Tau](./train_model.py)

---

### [16. Initialize Face Verification](./verification.py)

---

### [17. Embedding](./verification.py)

---

### [18. Verify](./verification.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)