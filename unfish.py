#! /usr/bin/env python3

import numpy as np
from cv2 import destroyAllWindows, getOptimalNewCameraMatrix, imshow, undistort, waitKey
from supervision import VideoInfo
from typer import run
from vidgear.gears import VideoGear, WriteGear

cam_mtx = np.load('cam_mtx.npy')
print('camera matrix:\n', cam_mtx)
dist_coefs = np.load('dist_coefs.npy')
print('distortion coefficients: ', dist_coefs)


def app(source: str, out: str = 'out.mp4'):
  stream = VideoGear(source=source).start()
  writer = WriteGear(out)
  w, h = VideoInfo.from_video_path(source).resolution_wh
  s = min(w, h)
  d = abs(w - h) // 2

  new_cam_mtx, roi = getOptimalNewCameraMatrix(cam_mtx, dist_coefs, (s, s), 1, (s, s))
  x, y, new_w, new_h = roi
  print(roi)
  while True:
    f = stream.read()
    if f is None:
      break
    if w != h:
      f = f[:, d : d + h] if w > h else f[d : d + w, :]
    f = undistort(
      f,
      cam_mtx,
      dist_coefs,
      None,
      new_cam_mtx,
    )[y : y + new_h, x : x + new_w]
    imshow('', f)
    if waitKey(1) & 0xFF == ord('q'):
      break
    writer.write(f)

  stream.stop()
  writer.close()
  destroyAllWindows()


if __name__ == '__main__':
  run(app)
