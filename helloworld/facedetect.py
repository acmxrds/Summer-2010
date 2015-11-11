## The face detection demo for XRDS Hello World.
## Adapted from OpenCV examples.
##
## Instructions: the detection will start automatically from
## an available camera. If more than one camera is present, change the
## index at line 16. Press ESC to quit.

import sys
import cv

storage = cv.CreateMemStorage(0)
image_scale = 1.3
haar_scale = 1.2
min_neighbors = 1
haar_flags = 0
camera_index = 0 	# can change index if more than one camera
cascade_name = "./haarcascade_frontalface_alt_tree.xml"

def detect_and_draw( img ):
	# allocate temporary images
	gray = cv.CreateImage( (img.width,img.height), 8, 1 )
	small_img = cv.CreateImage(
		(cv.Round(img.width/image_scale),
			cv.Round (img.height/image_scale)
		), 8, 1 )
		
	# convert color input image to grayscale
	cv.CvtColor( img, gray, cv.CV_BGR2GRAY )
	
	# scale input image for faster processing
	cv.Resize( gray, small_img, cv.CV_INTER_NN )
	cv.EqualizeHist( small_img, small_img )
	
	# start detection
	if( cascade ):
		faces = cv.HaarDetectObjects(
			small_img, cascade,
			storage, haar_scale,
			min_neighbors, haar_flags )
		if faces:
			for (x,y,w,h),n in faces:
				# the input to cvHaarDetectObjects was resized,
				# so scale the bounding box of each face
				# and convert it to two CvPoints
				pt1 = ( 
					int(x*image_scale),
					int(y*image_scale)
				)
				pt2 = (
					int((x+w)*image_scale),
					int((y+h)*image_scale)
				)
				# Draw the rectangle on the image
				cv.Rectangle( img, pt1, pt2,
					cv.CV_RGB(255,0,0), 3, 8, 0 )
	cv.ShowImage( "result", img )

if __name__ == '__main__':
	
	# Load the Haar cascade
	cascade = cv.Load(cascade_name)
	
	# Start capturing
	capture = cv.CaptureFromCAM(camera_index)
	
	# Create the output window
	cv.NamedWindow("result",1)

	frame_copy=None
	while True:
		frame = cv.QueryFrame( capture )
		
		# make a copy of the captured frame
		if not frame_copy:
			frame_copy = cv.CreateImage(
				(frame.width,frame.height),
				cv.IPL_DEPTH_8U,
				frame.nChannels
			)
		cv.Copy( frame, frame_copy )
		
		detect_and_draw(frame_copy)
		
		c = cv.WaitKey(7)
		if c==27: # Escape pressed
			break