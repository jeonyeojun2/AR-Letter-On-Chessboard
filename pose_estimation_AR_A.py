import numpy as np
import cv2 as cv

# Calibration results from Homework #3
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075,
                       -0.0004420196146339175, 0.0001149909868437517,
                       -0.01803978785585194])

# Input video
video = cv.VideoCapture('chessboard1.mp4')
assert video.isOpened(), 'Cannot open video file'

# Chessboard pattern
board_pattern = (10, 7)
board_cellsize = 0.025  # meters
criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Chessboard 3D points
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# AR object (3D points of alphabet 'A' in local coordinates)
A_shape = np.array([
    [0, 0, 0], [0.5, 1.5, 0], [1.0, 0, 0],  # base triangle
    [0.25, 0.75, 0], [0.75, 0.75, 0]       # crossbar of A
]) * board_cellsize + np.array([0.25, 2.5, -1]) * board_cellsize  # shift + height

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv.findChessboardCorners(img, board_pattern, criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Project the 3D A shape to 2D
        projected, _ = cv.projectPoints(A_shape, rvec, tvec, K, dist_coeff)
        projected = np.int32(projected).reshape(-1, 2)

        # Draw the "A" shape
        cv.line(img, projected[0], projected[1], (0, 0, 255), 2)
        cv.line(img, projected[1], projected[2], (0, 0, 255), 2)
        cv.line(img, projected[0], projected[2], (0, 0, 255), 2)
        cv.line(img, projected[3], projected[4], (255, 0, 0), 2)

        # 카메라 위치 정보 표시
        R, _ = cv.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()
        info = f'XYZ: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    cv.imshow('AR Pose Estimation - A', img)
    key = cv.waitKey(10)
    if key == ord(' '):  # Pause
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
