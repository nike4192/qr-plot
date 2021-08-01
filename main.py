import math
import cv2 as cv
import numpy as np
from pyzbar import pyzbar

cap = cv.VideoCapture(0)

objp = np.float32([
	[0, 0, 0],
	[0, 1, 0],
	[1, 1, 0],
	[1, 0, 0]])

axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)

mtx = np.asarray([
	[540.57702037,   0.,         377.44588566],
	[  0.,         535.61666116, 308.3818536 ],
	[  0.,           0.,           1.        ]])

dist = np.asarray([[ 1.91838272e-01, -1.38823625e+00, 7.93178983e-03, 3.01733229e-03, 3.79416310e+00]])

def calibrate_camera(objpoints, imgpoints, shape):
	global mtx, dist
	_, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)

def draw(img, corners, imgpts):

	# print(corners)
	corner = (int(corners[0][0][0]), int(corners[0][0][1]))
	# print(corner)
	img = cv.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)
	img = cv.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)
	img = cv.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)
		
	return img

def linear_to(oa, ob, na, nb, v):
	return ((v - oa) / (ob - oa)) * (nb - na) + na

def draw_graph(equation_string):

	min_v = -10
	max_v = 10

	width = 200
	height = 200
	graph_img = np.zeros((width, height, 3), np.uint8)
	graph_img.fill(255)

	ox = None
	oy = None

	for x in np.arange(min_v, max_v, 0.25):
		y = eval(equation_string)

		nx = linear_to(min_v, max_v, 0, 200, x)
		ny = linear_to(min_v, max_v, 0, 200, -y)

		if ox and oy:
			cv.line(graph_img, (int(ox), int(oy)), (int(nx), int(ny)), (0, 0, 0), 2)

		ox = nx
		oy = ny

	return graph_img

while True:

	if cv.waitKey(1) == 27:
		break

	ret, frame = cap.read()
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	if frame is None:
		print('--(!) No captured frame -- Break!')
		break

	# Detect and decode the qrcode

	barcodes = pyzbar.decode(frame_gray)

	if barcodes:
		for barcode in barcodes:
			
			objpoints = [objp]
			imgpoints = np.asarray([barcode.polygon], np.float32)

			p0 = imgpoints
			
			_, rvecs, tvecs = cv.solvePnP(objp, imgpoints[0], mtx, dist)
			imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

			rows, cols, ch = frame.shape

			img = frame

			try:
				graph = draw_graph(barcode.data)

				cv.imshow('graph', graph)

				graph_shape_pts = np.float32([[200,200], [0,200], [0,0], [200,0]])
				M = cv.getPerspectiveTransform(graph_shape_pts, imgpoints)

				dst = cv.warpPerspective(graph, M, (cols, rows))
				img = cv.add(frame, dst)

			except:
				pass

			img = draw(img, imgpoints, imgpts)
			font = cv.FONT_HERSHEY_SIMPLEX
			cv.putText(img, str(barcode.data), (int(cols / 2), int(rows * 2 / 3)), font, 1, (255, 255, 255), 2, cv.LINE_AA)
			cv.imshow('Results', img)

		if cv.waitKey(1) == ord('c'):
			calibrate_camera(objpoints, imgpoints, frame_gray.shape[::-1])

	cv.imshow('Results', frame)