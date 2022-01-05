import pykitti as kitti
import cv2 as cv

# Load the data
basedir = './dataset'
sequence = '04'
dataset = kitti.odometry(basedir, sequence, frames = range(0, 100))

