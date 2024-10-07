import matplotlib.pyplot as plt
import numpy as np

# define function which has an edge
x = np.linspace(-10, 10, 100)

y = (1 - np.exp(-2*x))/(1 + np.exp(-2*x))


plt.plot(x, y, color ='purple')
plt.xlabel('x')
plt.ylabel('y')
plt.title('example of edge in one dimension')

plt.show()

# first order derivative of the defined function
dy_dx = np.gradient(y, x)

plt.plot(x, dy_dx, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1st order derivative')

plt.show()

# second order derivative of the defined function
dy_dx2 = np.gradient(dy_dx, x)

plt.plot(x, dy_dx2, color = "green")
plt.xlabel('x')
plt.ylabel('y')
plt.title('2nd order derivative')
plt.show()
