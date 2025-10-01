#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 2
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		#Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000/self._cam_fps)

		self.sift = cv2.SIFT_create()
		self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

		self.kp_template = None
		self.des_template = None


	
	
	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		print("Loaded template image file: " + self.template_path)

		# Load grayscale for feature extraction
		template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		self.kp_template, self.des_template = self.sift.detectAndCompute(template_img, None)
		self.template_gray = template_img
		self._is_template_loaded = True

		# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		#TODO run SIFT on the captured frame
		if not ret:
			return
		
		if self._is_template_loaded:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			kp_frame, des_frame = self.sift.detectAndCompute(gray, None) # find key points andcamera frame descriptors for the 

			# If descriptors for both exist
			if des_frame is not None and self.des_template is not None:
				matches = self.flann.knnMatch(self.des_template, des_frame, k=2) # compare reference and camera descriptors
				# setting k to 2, we find the two closest matches for each descriptor, used to apply ratio test

				# Apply "Lowe's ratio test"
				# Basically if the best match is way better than the second best, then it's probably actually a good match
				# throw out the rest
				good_matches = []
				for m, n in matches:
					if m.distance < 0.7 * n.distance:
						good_matches.append(m)

				match_img = cv2.drawMatches(
							self.template_gray, self.kp_template,
							gray, kp_frame,
							good_matches, None,
							flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
						)
				cv2.imshow("Feature Matches", match_img)

				# If enough good matches are found, then we can try draw a homography
				if len(good_matches) > 10:
					src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
					dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

					H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
					if H is not None:
						h, w = self.template_gray.shape
						corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
						transformed_corners = cv2.perspectiveTransform(corners, H)
						frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0,255,0), 3)
					else:
						# No homography: draw keypoints
						frame = cv2.drawKeypoints(frame, kp_frame, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
						
				else:
					# Not enough matches: draw keypoints
					frame = cv2.drawKeypoints(frame, kp_frame, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


		pixmap = self.convert_cv_to_pixmap(frame)
		self.live_image_label.setPixmap(pixmap)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
