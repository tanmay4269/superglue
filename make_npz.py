from video_stitch_help import *

video = cv.VideoCapture('videos/vid1.MOV')

frames = []
resize_factor = 0.5  

while True:
  ret, frame = video.read()
  if not ret:
      break
  
  frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  rotated_frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
  target_frame = cv.resize(rotated_frame, None, fx=resize_factor, fy=resize_factor)
  frames.append(target_frame)

  
video.release()

frames = np.array(frames)
np.save("npz_files/vid1", frames)