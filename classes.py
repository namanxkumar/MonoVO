import numpy as np
import cv2 as cv

MIN_FEATURES = 2000


def featureTracking(prevImage, currImage, prevPts):
    # Initialize the parameters
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS |
                               cv.TERM_CRITERIA_COUNT, 30, 0.01),
                     flags=0,
                     minEigThreshold=1e-4)
    currPts, st, err = cv.calcOpticalFlowPyrLK(
        prevImage, currImage, prevPts, None, **lk_params)

    # Select good points
    st = st.reshape(st.shape[0])
    kp1 = prevPts[st == 1]
    currPts = currPts[st == 1]

    return kp1, currPts

class Camera:
    def __init__(self, width, height, fx, fy, cx, cy, k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0):
        # Camera intrinsics

        self.width = width
        self.height = height
            # Focal Lengths
        self.fx = fx
        self.fy = fy
            # Principal Point
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3] # distortion matrices

    def project(self, point):
        """Project 3D point to 2D image point"""
        x = (point[0] * self.fx / point[2]) + self.cx
        y = (point[1] * self.fy / point[2]) + self.cy
        return np.array([x, y])
    
    def unproject(self, point):
        """Project 2D image point to 3D point"""
        x = (point[0] - self.cx) * point[2] / self.fx
        y = (point[1] - self.cy) * point[2] / self.fy
        z = point[2]
        return np.array([x, y, z])
    
    def distort(self, point):
        """Distort 2D image point"""
        if self.distortion:
            x = point[0]
            y = point[1]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            x_d = x * (1 + self.d[0] * r2 + self.d[1] * r4 + self.d[4] * r6) + 2 * self.d[2] * x * y + self.d[3] * (r2 + 2 * x * x)
            y_d = y * (1 + self.d[0] * r2 + self.d[1] * r4 + self.d[4] * r6) + self.d[2] * (r2 + 2 * y * y) + 2 * self.d[3] * x * y
            return np.array([x_d, y_d])
        else:
            return point

    def undistort(self, point):
        """Undistort 2D image point"""
        if self.distortion:
            x = point[0]
            y = point[1]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            x_u = x - 2 * self.d[2] * x * y + self.d[3] * (r2 + 2 * x * x) + self.d[0] * r2 * 2 + self.d[1] * r4 * 2 + self.d[4] * r6 * 2
            y_u = y - self.d[2] * (r2 + 2 * y * y) + 2 * self.d[3] * x * y + self.d[0] * r2 * 2 + self.d[1] * r4 * 2 + self.d[4] * r6 * 2
            return np.array([x_u, y_u])
        else:
            return point

    def projectPoints(self, points):
        """Project 3D points to 2D image points"""
        return [self.project(point) for point in points]
    
    def unprojectPoints(self, points):
        """Project 2D image points to 3D points"""
        return [self.unproject(point) for point in points]
    
    def distortPoints(self, points):
        """Distort 2D image points"""
        return [self.distort(point) for point in points]
    
    def undistortPoints(self, points):
        """Undistort 2D image points"""
        return [self.undistort(point) for point in points]
    
class VisualOdometry:
    def __init__(self, camera, dataset):
        self.frameState = 0 # Frame State
        self.camera = camera # Camera model
        self.currFrame = None # Current frame
        self.prevFrame = None # Previous frame
        self.currR = None # Current rotation
        self.currt = None # Current translation
        self.prevPts = None # Previous points
        self.currPts = None # Current points
        self.focal = camera.fx # Focal length
        self.pp = (camera.cx, camera.cy) # Principal point
        self.dataset = dataset # Dataset
        self.poses = dataset.oxts # List of poses
        self.gtX, self.gtY, self.gtZ = 0, 0, 0 # Ground truth position
        self.detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def getAbsoluteScale(self, frameId):
        """Get absolute scale of current frame"""
        pose = np.array(self.dataset.oxts[frameId-1].T_w_imu).dot(np.linalg.inv(np.array(self.dataset.calib.T_cam0_imu)))
        """
            T_w_imu is of the form:
            [
            [r11, r12, r13, t1]
            [r21, r22, r23, t2]
            [r31, r32, r33, t3]
            [  0,   0,   0,  1]
            ]
        """
        x_prev = pose[0, 3]
        y_prev = pose[1, 3]
        z_prev = pose[2, 3]

        pose = np.array(self.dataset.oxts[frameId].T_w_imu).dot(np.linalg.inv(np.array(self.dataset.calib.T_cam0_imu)))
        x_curr = pose[0, 3]
        y_curr = pose[1, 3]
        z_curr = pose[2, 3]
        scale = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2 + (z_curr - z_prev)**2)
        self.gtX, self.gtY, self.gtZ = x_curr, y_curr, z_curr
        return scale

    def processFirstFrame(self):
        """Process first frame"""
        self.prevPts = self.detector.detect(self.currFrame) # Detect features in first frame
        self.prevPts = np.array([x.pt for x in self.prevPts], dtype=np.float32) # Convert Keypoint to Point2f
        self.frameState = 1 # Increment frame count

    def processSecondFrame(self):
        """Process second frame"""
        self.prevPts, self.currPts = featureTracking(self.prevFrame, self.currFrame, self.prevPts) # Track features in second frame
        E, mask = cv.findEssentialMat(self.currPts, self.prevPts, focal=self.focal, pp=self.pp, method=cv.RANSAC, prob=0.999, threshold=1.0) # Compute essential matrix
        _, self.currR, self.currt, mask = cv.recoverPose(E, self.currPts, self.prevPts, focal=self.focal, pp=self.pp) # Recover pose
        self.frameState = 2 # Increment frame count
        self.prevPts = self.currPts # Update previous points
    
    def processFrame(self, frame_id):
        self.prevPts, self.currPts = featureTracking(self.prevFrame, self.currFrame, self.prevPts) # Track features in second frame
        E, mask = cv.findEssentialMat(self.currPts, self.prevPts, focal=self.focal, pp=self.pp, method=cv.RANSAC, prob=0.999, threshold=1.0) # Compute essential matrix
        _, R, t, mask = cv.recoverPose(E, self.currPts, self.prevPts, focal=self.focal, pp=self.pp) # Recover pose
        absScale = self.getAbsoluteScale(frame_id) # Get absolute scale
        if(absScale > 0.1):
            self.currt = self.currt + absScale * self.currR.dot(t) # Update translation
            self.currR = R.dot(self.currR) # Update rotation
        if(self.currPts.shape[0] < MIN_FEATURES):
            self.currPts = self.detector.detect(self.currFrame) # Replenish features in frame
            self.currPts = np.array([x.pt for x in self.currPts], dtype=np.float32) # Convert Keypoint to Point2f
        self.prevPts = self.currPts # Update previous points

    def update(self, img, frame_id):
        assert(img.ndim == 2 and img.shape[0] == self.camera.height and img.shape[1] == self.camera.width), "Frame: provided image does not have the same size as the camera model, or image is not grayscale"
        self.currFrame = img # Copy image
        if(self.frameState == 2):
            self.processFrame(frame_id)
        elif(self.frameState == 1):
            self.processSecondFrame()
        elif(self.frameState == 0):
            self.processFirstFrame()
        self.prevFrame = self.currFrame # Update previous frame
