import pykitti as kitti
import cv2 as cv
import numpy as np

# Load the data
basedir = './dataset'
date = '2011_09_26'
drive = '0093'
dataset = kitti.raw(basedir, date, drive)

from classes import Camera, VisualOdometry

camera = Camera(height=375, width=1242, fx=7.215377e+02, fy=7.215377e+02, cx=6.095593e+02, cy=1.728540e+02)
vo = VisualOdometry(camera, dataset)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

for img_id in range(len(dataset.cam0_files)):
    img = np.array(dataset.get_gray(img_id)[0])
    vo.update(img, img_id)

    currt = vo.currt

    if(img_id > 2):
        x, y, z = currt[0], currt[1], currt[2]
    else:
        x, y, z = 0, 0, 0
    
    draw_x, draw_y = int(-x) + 290, int(z) + 90
    true_x, true_y = int(-vo.gtX) + 290, int(-vo.gtZ) + 90
    cv.circle(traj, (draw_x, draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 2)
    # cv.circle(traj, (true_x, true_y), 1, (255,255,255), 2)
    cv.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv.putText(traj, text, (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 8)

    cv.imshow('Road cam', img)
    cv.imshow('Trajectory', traj)
    cv.waitKey(1)

cv.imwrite('map.png', traj)