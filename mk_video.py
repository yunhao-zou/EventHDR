import cv2
import os
import glob
import numpy as np

fps = 15    #FPS
size=(320*3, 240)  # video size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result_dir = 'reconstruction/'
scenes = os.listdir(result_dir)
scenes = [scene for scene in scenes if os.path.isdir(scene)]
for scene in scenes:
    videoWriter = cv2.VideoWriter('{}/{}.mp4'.format(result_dir, scene), fourcc, fps, size, True)
    event_dir = sorted(glob.glob('{}/{}/events_*.png'.format(result_dir, scene)))[::10]
    for event_file in event_dir:
        print(event_file + ' done!')
        event = cv2.imread(event_file)
        frame = cv2.imread(event_file.replace('events_', 'recon_'))
        ref = cv2.imread(event_file.replace('events_', 'ref_'))
        video = np.concatenate([event, frame, ref], axis=1)
        print(video.shape, event.shape, np.mean(event))
        # frame = np.array(frame, dtype=np.float)
        videoWriter.write(video.astype(np.uint8))
        print(event_file + ' done!')
    videoWriter.release()