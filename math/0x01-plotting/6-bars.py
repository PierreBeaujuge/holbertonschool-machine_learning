#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

apples = plt.bar([1, 2, 3], fruit[0], 0.5, color='red')
bananas = plt.bar([1, 2, 3], fruit[1], 0.5, bottom=fruit[0], color='yellow')
oranges = plt.bar([1, 2, 3], fruit[2], 0.5,
                  bottom=fruit[0] + fruit[1], color='#ff8000')
peaches = plt.bar([1, 2, 3], fruit[3], 0.5,
                  bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4')

plt.xlim(0.6, 3.4, 1)
plt.xticks(np.arange(1, 4, 1), ('Farrah', 'Fred', 'Felicia'))
plt.ylim(0, 80, 10)
plt.xlabel('')
plt.ylabel('Quantity of Fruit')
plt.legend((apples[0], bananas[0], oranges[0], peaches[0]),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.title('Number of Fruit per Person')

plt.show()
