from ising import update, update_wolff
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import numpy as np

def animate_frame(i):
    global grid
    grid = update(grid, temp)
    img.set_data(grid)
    return [img]

temp = 2.236 # Critical temperature for 2D Ising Model [Onsager solution]
L = 400 # Size of lattice

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

grid = 2 * np.random.randint(0, 2, size=(L, L), dtype=int) - 1
img = ax.matshow(grid, cmap='Reds')

anim = animation.FuncAnimation(fig, animate_frame, frames=300, interval=50, blit=True)

temp_slider = Slider(
    ax=plt.axes([0.25, 0.1, 0.65, 0.03]),
    label='Temperature', valmin=0.02, valmax=5, valinit=temp
)

def slider_upd(val): 
    global temp 
    temp = temp_slider.val

temp_slider.on_changed(slider_upd)

plt.show()
