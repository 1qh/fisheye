#! /usr/bin/env python3

from time import perf_counter

import numpy as np
from cv2 import (
  destroyAllWindows,
  getOptimalNewCameraMatrix,
  imread,
  imshow,
  imwrite,
  undistort,
  waitKey,
)
from numpy import ndarray
from supervision import VideoInfo
from typer import run
from vidgear.gears import VideoGear, WriteGear

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')


class FisheyeFlatten:
  def __init__(
    self,
    reso: tuple[int, int],
    aspect_ratio: float | None = 16 / 9,
  ):
    w, h = reso
    s = min(w, h)
    d = abs(w - h) // 2
    self.camera_matrix = camera_matrix
    self.dist_coeffs = dist_coeffs
    self.new_camera_matrix, roi = getOptimalNewCameraMatrix(
      camera_matrix, dist_coeffs, (s, s), 1, (s, s)
    )
    right, top, new_w, new_h = roi
    bottom = top + new_h
    left = right + new_w

    if aspect_ratio:
      if new_w / new_h > aspect_ratio:
        spare = (new_w - int(new_h * aspect_ratio)) // 2
        left += spare
        right -= spare
      else:
        spare = (new_h - int(new_w / aspect_ratio)) // 2
        top += spare
        bottom -= spare

    self.crop = slice(top, bottom), slice(right, left)
    self.slic = (
      (slice(None), slice(None))
      if w == h
      else ((slice(None), slice(d, d + h)) if w > h else (slice(d, d + w), slice(None)))
    )

  def __call__(self, f: ndarray) -> ndarray:
    return undistort(
      f[self.slic],
      self.camera_matrix,
      self.dist_coeffs,
      None,
      self.new_camera_matrix,
    )[self.crop]


def app(source: str, out: str = 'out'):
  if source.endswith('.mp4'):
    export_vid(source, out)
  elif source.endswith('.jpg'):
    img = imread(source)
    flatten = FisheyeFlatten(img.shape[:2][::-1], None)

    start = perf_counter()
    img = flatten(img)
    print(f'Taken: {perf_counter() - start}s')

    imwrite(f'{out}.jpg', (img))


def export_vid(source, out):
  stream = VideoGear(source=source).start()
  writer = WriteGear(f'{out}.mp4')
  reso = VideoInfo.from_video_path(source).resolution_wh
  flatten = FisheyeFlatten(reso)

  while (img := stream.read()) is not None:
    img = flatten(img)
    imshow('', img)
    if waitKey(1) & 0xFF == ord('q'):
      break
    writer.write(img)

  stream.stop()
  writer.close()
  destroyAllWindows()


if __name__ == '__main__':
  run(app)
