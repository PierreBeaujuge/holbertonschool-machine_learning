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

Complete the following source code to plot `y` as a line graph:

- `y` should be plotted as a solid red line
- The x-axis should range from 0 to 10

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
```
![](./0x01-images/img_1.png)

---

### [1. Scatter](./1-scatter.py)

Complete the following source code to plot `x ↦ y` as a scatter plot:

- The x-axis should be labeled `Height (in)`
- The y-axis should be labeled `Weight (lbs)`
- The title should be `Men's Height vs Weight`
- The data should be plotted as magenta points

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here
```
![](./0x01-images/img_2.png)

---

### [2. Change of scale](./2-change_scale.py)

Complete the following source code to plot `x ↦ y` as a line graph:

- The x-axis should be labeled `Time (years)`
- The y-axis should be labeled `Fraction Remaining`
- The title should be `Exponential Decay of C-14`
- The y-axis should be logarithmically scaled
- The x-axis should range from 0 to 28650

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here
```
![](./0x01-images/img_3.png)

---

### [3. Two is better than one](./3-two.py)

Complete the following source code to plot `x ↦ y1` and `x ↦ y2` as line graphs:

- The x-axis should be labeled `Time (years)`
- The y-axis should be labeled `Fraction Remaining`
- The title should be Exponential `Decay of Radioactive Elements`
- The x-axis should range from 0 to 20,000
- The y-axis should range from 0 to 1
- `x ↦ y1` should be plotted with a dashed red line
- `x ↦ y2` should be plotted with a solid green line
- A legend labeling `x ↦ y1` as `C-14` and `x ↦ y2` as `Ra-226` should be placed in the upper right hand corner of the plot

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here
```
![](./0x01-images/img_4.png)

---

### [4. Frequency](./4-frequency.py)

Complete the following source code to plot a histogram of student scores for a project:

- The x-axis should be labeled `Grades`
- The y-axis should be labeled `Number of Students`
- The x-axis should have bins every 10 units
- The title should be `Project A`
- The bars should be outlined in black

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
```
![](./0x01-images/img_5.png)

---

### [5. All in One](./5-all_in_one.py)

Complete the following source code to plot all 5 previous graphs in one figure:

- All axis labels and plot titles should have a font size of `x-small` (to fit nicely in one figure)
- The plots should make a 3 x 2 grid
- The last plot should take up two column widths (see below)
- The title of the figure should be `All in One`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
```
![](./0x01-images/img_6.png)

---

### [6. Stacking Bars](./6-bars.py)

Complete the following source code to plot a stacked bar graph:

- `fruit` is a matrix representing the number of fruit various people possess
  - The columns of `fruit` represent the number of fruit `Farrah`, `Fred`, and `Felicia` have, respectively
  - The rows of `fruit` represent the number of `apples`, `bananas`, `oranges`, and `peaches`, respectively
- The bars should represent the number of fruit each person possesses:
  - The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
  - Each fruit should be represented by a specific color:
    - `apples` = red
    - `bananas` = yellow
    - `oranges` = orange (`#ff8000`)
    - `peaches` = peach (`#ffe5b4`)
    - A legend should be used to indicate which fruit is represented by each color
  - The bars should be stacked in the same order as the rows of `fruit`, from bottom to top
  - The bars should have a width of `0.5`
- The y-axis should be labeled `Quantity of Fruit`
- The y-axis should range from 0 to 80 with ticks every 10 units
- The title should be `Number of Fruit per Person`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
```
![](./0x01-images/img_7.png)

---

### [7. Gradient](./100-gradient.py)

Complete the following source code to create a scatter plot of sampled elevations on a mountain:

- The x-axis should be labeled `x coordinate (m)`
- The y-axis should be labeled `y coordinate (m)`
- The title should be `Mountain Elevation`
- A colorbar should be used to display elevation
- The colorbar should be labeled `elevation (m)`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here
```
![](./0x01-images/img_8.png)

---

### [8. PCA](./101-pca.py)

Principle Component Analysis (PCA) is a vital procedure used in data science for reducing the dimensionality of data (in turn, decreasing computation cost). It is also largely used for visualizing high dimensional data in 2 or 3 dimensions. For this task, you will be visualizing the `Iris flower data set`. You will need to download the file `pca.npz` to test your code. You do not need to push this dataset to github. Complete the following source code to visualize the data in 3D:

```
#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here
```

- The title of the plot should be `PCA of Iris Dataset`
- `data` is a `np.ndarray` of shape `(150, 4)`
  - `150` => the number of flowers
  - `4` => petal length, petal width, sepal length, sepal width
- `labels` is a `np.ndarray` of shape `(150,)` containing information about what species of iris each data point represents:
  - `0` => Iris Setosa
  - `1` => Iris Versicolor
  - `2` => Iris Virginica
- `pca_data` is a `np.ndarray` of shape `(150, 3)`
  - The columns of `pca_data` represent the 3 dimensions of the reduced data, i.e., x, y, and z, respectively
- The x, y, and z axes should be labeled `U1`, `U2`, and `U3`, respectively
- The data points should be colored based on their labels using the `plasma` color map

![](./0x01-images/img_9.png)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)