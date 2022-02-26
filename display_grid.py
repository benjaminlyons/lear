import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

imgs = []
directory = "fakes"
for filename in os.listdir(directory):
    print(filename)
    image = PIL.Image.open(os.path.join(directory, filename))
    imgs.append(np.array(image))

# for row in range(9):
#     for col in range(9):
#         image = PIL.Image.open(f"grids/dim_{row}_{col}.jpg")
#         imgs.append(np.array(image))
#
fig = plt.figure(figsize=(9,9))
grid = ImageGrid(fig, 111,
        nrows_ncols=(9,9),
        axes_pad=0.1,
        )

for ax, im in zip(grid, imgs):
    ax.imshow(im)

plt.savefig("grid.jpg")
