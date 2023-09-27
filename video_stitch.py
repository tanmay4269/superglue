from video_stitch_help import *

frames = np.load("npz_files/vid1.npy")

frames = frames[9*30:18*30:2]

img = generate_pano(frames[1], frames[0])

delay_time = 1 # ms

for frame in frames:
  try:
    img = generate_pano(frame, img)
    cv.imshow("img", img)
  except:
    pass

  if cv.waitKey(delay_time):
      break

cv.waitKey(0)
cv.destroyAllWindows()
    
