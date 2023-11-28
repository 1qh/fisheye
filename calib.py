import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from cv2 import (
  COLOR_GRAY2BGR,
  IMREAD_GRAYSCALE,
  TERM_CRITERIA_COUNT,
  TERM_CRITERIA_EPS,
  calibrateCamera,
  cornerSubPix,
  cvtColor,
  drawChessboardCorners,
  findChessboardCorners,
  imread,
  imwrite,
  resize,
)

threads_num = 12
img_mask = 'frames/*.jpg'
vis_dir = './debug'
pattern_size = (5, 8)


img_names = glob(img_mask)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
obj_points = []
img_points = []
h, w = imread(img_names[0], IMREAD_GRAYSCALE).shape[:2]
print('h =', h, '\nw =', w)


def splitfn(fn):
  path, fn = os.path.split(fn)
  name, ext = os.path.splitext(fn)
  return path, name, ext


def calib(fn):
  img = imread(fn, IMREAD_GRAYSCALE)
  img = resize(img, (w, h))
  if img is None:
    print('Failed to load', fn)
    return None
  found, corners = findChessboardCorners(img, pattern_size)
  if found:
    term = (TERM_CRITERIA_EPS + TERM_CRITERIA_COUNT, 30, 0.1)
    cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if vis_dir:
      vis = cvtColor(img, COLOR_GRAY2BGR)
      drawChessboardCorners(vis, pattern_size, corners, found)
      name = splitfn(fn)[1]
      outfile = os.path.join(vis_dir, name + '.jpg')
      imwrite(outfile, vis)
  if not found:
    print(fn)
    return None
  print(fn, 'OK')
  return (corners.reshape(-1, 2), pattern_points)


# chessboards = [calib(x) for x in img_names]
pool = Pool(threads_num)
chessboards = pool.map(calib, img_names)
chessboards = [x for x in chessboards if x]
print(len(chessboards), 'chessboards found')
for corners, pattern_points in chessboards:
  img_points.append(corners)
  obj_points.append(pattern_points)

rms, cam_mtx, dist_coefs, _, _ = calibrateCamera(
  obj_points,
  img_points,
  (w, h),
  None,
  None,
)
print('\nRMS:', rms)
print('camera matrix:\n', cam_mtx)
print('distortion coefficients: ', dist_coefs)

np.save('cam_mtx.npy', cam_mtx)
np.save('dist_coefs.npy', dist_coefs)
