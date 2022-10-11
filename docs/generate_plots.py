import numpy as np
import matplotlib.pyplot as plt
from delicatessen.utilities import robust_loss_functions

# Creating a range of values for the robust-loss function plots

plt.figure(figsize=(7, 5))
plt.subplot(221)
x = np.linspace(-4, 4, 200)
plt.plot(x, robust_loss_functions(x, loss='huber', k=1), '-', color='red', label='k=1')
plt.plot(x, robust_loss_functions(x, loss='huber', k=2), '-', color='green', label='k=2')
plt.plot(x, robust_loss_functions(x, loss='huber', k=3), '-', color='blue', label='k=3')
plt.title("Huber")
plt.ylabel(r"$f_k(x)$")
plt.xlabel(r"$x$")
plt.legend()

plt.subplot(222)
x = np.linspace(-4, 4, 200)
plt.plot(x, robust_loss_functions(x, loss='tukey', k=1), '-', color='red', label='k=1')
plt.plot(x, robust_loss_functions(x, loss='tukey', k=2), '-', color='green', label='k=2')
plt.plot(x, robust_loss_functions(x, loss='tukey', k=3), '-', color='blue', label='k=3')
plt.title("Tukey's Biweight")
plt.xlabel(r"$x$")

plt.subplot(223)
x = np.linspace(-12, 12, 200)
plt.plot(x, robust_loss_functions(x, loss='andrew', k=1), '-', color='red')
plt.plot(x, robust_loss_functions(x, loss='andrew', k=2), '-', color='green')
plt.plot(x, robust_loss_functions(x, loss='andrew', k=3), '-', color='blue')
plt.title("Andrew's Sine")
plt.ylabel(r"$f_k(x)$")
plt.xlabel(r"$x$")

plt.subplot(224)
x = np.linspace(-4, 4, 200)
plt.plot(x, robust_loss_functions(x, loss='hampel', k=1, a=1/3, b=2/3), '-', color='red')
plt.plot(x, robust_loss_functions(x, loss='hampel', k=2, a=2/3, b=4/3), '-', color='green')
plt.plot(x, robust_loss_functions(x, loss='hampel', k=3, a=3/3, b=6/3), '-', color='blue')
plt.title("Hampel")
plt.xlabel(r"$x$")

plt.tight_layout()
plt.savefig("images/robust_loss.png", format='png', dpi=300)
plt.close()
