# 0x03. Probability

## Learning Objectives

- Basic probability notation
- What is independence? What is disjoint?
- What is a union? intersection?
- What are the general addition and multiplication rules?
- What is a probability distribution?
- What is a probability distribution function? probability mass function?
- What is a cumulative distribution function?
- What is a percentile?
- What is mean, standard deviation, and variance?
- Common probability distributions

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
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

## Tasks

### [0. Initialize Poisson](./poisson.py)

Create a class `Poisson` that represents a poisson distribution:

*   Class contructor `def __init__(self, data=None, lambtha=1.):`
    *   `data` is a list of the data to be used to estimate the distribution
    *   `lambtha` is the expected number of occurences in a given time frame
    *   Sets the instance attribute `lambtha`
        *   Saves `lambtha` as a float
    *   If `data` is not given, (i.e. `None`):
        *   Use the given `lambtha`
        *   If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    *   If `data` is given:
        *   Calculate the `lambtha` of `data`
        *   If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        *   If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`  

```
    alexa@ubuntu-xenial:0x03-probability$ ./0-main.py 
    Lambtha: 4.84
    Lambtha: 5.0
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [1. Poisson PMF](./poisson.py)

Update the class `Poisson`:

*   Instance method `def pmf(self, k):`
    *   Calculates the value of the PMF for a given number of “successes”
    *   `k` is the number of “successes”
        *   If `k` is not an integer, convert it to an integer
        *   If `k` is out of range, return `0`
    *   Returns the PMF value for `k`

```
    alexa@ubuntu-xenial:0x03-probability$ ./1-main.py 
    P(9): 0.03175849616802446
    P(9): 0.036265577412911795
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [2. Poisson CDF](./poisson.py)

Update the class `Poisson`:

*   Instance method `def cdf(self, k):`
    *   Calculates the value of the CDF for a given number of “successes”
    *   `k` is the number of “successes”
        *   If `k` is not an integer, convert it to an integer
        *   If `k` is out of range, return `0`
    *   Returns the CDF value for `k`

```
    alexa@ubuntu-xenial:0x03-probability$ ./2-main.py 
    F(9): 0.9736102067423525
    F(9): 0.9681719426208609
    alexa@ubuntu-xenial:0x03-probability$ 
```

---

### [3. Initialize Exponential](./exponential.py)

Create a class `Exponential` that represents an exponential distribution:

*   Class contructor `def __init__(self, data=None, lambtha=1.):`
    *   `data` is a list of the data to be used to estimate the distribution
    *   `lambtha` is the expected number of occurences in a given time frame
    *   Sets the instance attribute `lambtha`
        *   Saves `lambtha` as a float
    *   If `data` is not given (i.e. `None`):
        *   Use the given `lambtha`
        *   If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    *   If `data` is given:
        *   Calculate the `lambtha` of `data`
        *   If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        *   If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

```
    alexa@ubuntu-xenial:0x03-probability$ ./3-main.py 
    Lambtha: 2.1771114730906937
    Lambtha: 2.0
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [4. Exponential PDF](./exponential.py)

Update the class `Exponential`:

*   Instance method `def pdf(self, x):`
    *   Calculates the value of the PDF for a given time period
    *   `x` is the time period
    *   Returns the PDF value for `x`
    *   If `x` is out of range, return `0`

```
    alexa@ubuntu-xenial:0x03-probability$ ./4-main.py 
    f(1): 0.24681591903431568
    f(1): 0.2706705664650693
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [5. Exponential CDF](./exponential.py)

Update the class `Exponential`:

*   Instance method `def cdf(self, x):`
    *   Calculates the value of the CDF for a given time period
    *   `x` is the time period
    *   Returns the CDF value for `x`
    *   If `x` is out of range, return `0`

```
    alexa@ubuntu-xenial:0x03-probability$ ./5-main.py 
    F(1): 0.886631473819791
    F(1): 0.8646647167674654
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [6. Initialize Normal](./normal.py)

Create a class `Normal` that represents a normal distribution:

*   Class contructor `def __init__(self, data=None, mean=0., stddev=1.):`
    *   `data` is a list of the data to be used to estimate the distribution
    *   `mean` is the mean of the distribution
    *   `stddev` is the standard deviation of the distribution
    *   Sets the instance attributes `mean` and `stddev`
        *   Saves `mean` and `stddev` as floats
    *   If `data` is not given (i.e. `None`)
        *   Use the given `mean` and `stddev`
        *   If `stddev` is not a positive value, raise a `ValueError` with the message `stddev must be a positive value`
    *   If `data` is given:
        *   Calculate the mean and standard deviation of `data`
        *   If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        *   If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

```
    alexa@ubuntu-xenial:0x03-probability$ ./6-main.py 
    Mean: 70.59808015534485 , Stddev: 10.078822447165797
    Mean: 70.0 , Stddev: 10.0
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [7. Normalize Normal](./normal.py)

Update the class `Normal`:

*   Instance method `def z_score(self, x):`
    *   Calculates the z-score of a given x-value
    *   `x` is the x-value
    *   Returns the z-score of `x`
*   Instance method `def x_value(self, z):`
    *   Calculates the x-value of a given z-score
    *   `z` is the z-score
    *   Returns the x-value of `z`

```
    alexa@ubuntu-xenial:0x03-probability$ ./7-main.py 
    Z(90): 1.9250185174272068
    X(2): 90.75572504967644
    
    Z(90): 2.0
    X(2): 90.0
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [8. Normal PDF](./normal.py)

Update the class `Normal`:

*   Instance method `def pdf(self, x):`
    *   Calculates the value of the PDF for a given x-value
    *   `x` is the x-value
    *   Returns the PDF value for `x`

```
    alexa@ubuntu-xenial:0x03-probability$ ./8-main.py 
    PSI(90): 0.006206096804434349
    PSI(90): 0.005399096651147344
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [9. Normal CDF](./normal.py)

Update the class `Normal`:

*   Instance method `def cdf(self, x):`
    *   Calculates the value of the CDF for a given x-value
    *   `x` is the x-value
    *   Returns the CDF value for `x`

```
    alexa@ubuntu-xenial:0x03-probability$ ./9-main.py 
    PHI(90): 0.982902011086006
    PHI(90): 0.9922398930667251
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [10. Initialize Binomial](./binomial.py)

Create a class `Binomial` that represents a binomial distribution:

*   Class contructor `def __init__(self, data=None, n=1, p=0.5):`
    *   `data` is a list of the data to be used to estimate the distribution
    *   `n` is the number of Bernoulli trials
    *   `p` is the probability of a “success”
    *   Sets the instance attributes `n` and `p`
        *   Saves `n` as an integer and `p` as a float
    *   If `data` is not given (i.e. `None`)
        *   Use the given `n` and `p`
        *   If `n` is not a positive value, raise a `ValueError` with the message `n must be a positive value`
        *   If `p` is not a valid probability, raise a `ValueError` with the message `p must be greater than 0 and less than 1`
    *   If `data` is given:
        *   Calculate `n` and `p` from `data`
        *   Round `n` to the nearest integer
        *   _Hint: Calculate `p` first and then calculate `n`. Then recalculate `p`. Think about why you would want to do it this way?_
        *   If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        *   If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

```
    alexa@ubuntu-xenial:0x03-probability$ ./10-main.py 
    n: 50 p: 0.606
    n: 50 p: 0.6
    alexa@ubuntu-xenial:0x03-probability$ 
```

---

### [11. Binomial PMF](./binomial.py)

Update the class `Binomial`:

*   Instance method `def pmf(self, k):`
    *   Calculates the value of the PMF for a given number of “successes”
    *   `k` is the number of “successes”
        *   If `k` is not an integer, convert it to an integer
        *   If `k` is out of range, return `0`
    *   Returns the PMF value for `k`

```
    alexa@ubuntu-xenial:0x03-probability$ ./11-main.py 
    P(30): 0.11412829839570347
    P(30): 0.114558552829524
    alexa@ubuntu-xenial:0x03-probability$
```

---

### [12. Binomial CDF](./binomial.py)

Update the class `Binomial`:

*   Instance method `def cdf(self, k):`
    *   Calculates the value of the CDF for a given number of “successes”
    *   `k` is the number of “successes”
        *   If `k` is not an integer, convert it to an integer
        *   If `k` is out of range, return `0`
    *   Returns the CDF value for `k`
    *   _Hint: use the `pmf` method_

```
    alexa@ubuntu-xenial:0x03-probability$ ./12-main.py 
    F(30): 0.5189392017296368
    F(30): 0.5535236207894576
    alexa@ubuntu-xenial:0x03-probability$
```

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)