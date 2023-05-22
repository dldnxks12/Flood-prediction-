import sys
import imageio

filenames = []
for i in range(0, 51):
    filenames.append(str(i)+"_interpolation.png")

images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('interpolation.gif', images, "GIF", duration = 0.05)
