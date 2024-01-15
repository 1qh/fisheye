import cv2
import numpy as np


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
  h, w, _ = img.shape
  rows, cols = grid_shape
  dy, dx = h / rows, w / cols

  for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
    x = int(round(x))
    cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

  for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
    y = int(round(y))
    cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

  return img


im = cv2.imread('pcn.jpg')
im = draw_grid(im, (5, 5), thickness=1)
cv2.imwrite('pcn_grid.jpg', im)
