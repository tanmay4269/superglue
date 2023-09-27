import numpy as np
import cv2 as cv

frames = np.load("npz_files/vid1.npy")

frames = frames[9*30:18*30]

delay_time = 1 # ms
for i in range(frames.shape[0]):
  cv.imshow('img', frames[i])

  if cv.waitKey(delay_time) & 0xFF == ord('q'):
      break

cv.waitKey(0)
cv.destroyAllWindows()