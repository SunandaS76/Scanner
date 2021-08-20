#import

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as numpy
import argparse
import cv2
import imutils

# construct the arg parser 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)


# convert the image to grayscale, blur it, and find edges
# in the image
#print("image shape", image.shape[1])

y_end = image.shape[0]
x_end = image.shape[1]


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

flag = False
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		print("flag yes")
		flag = True
		break

if flag == False :
	screenCnt = numpy.array([[[0,0]] ,[[x_end,0]] ,[[x_end, y_end]] ,[[0 ,y_end]]])
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#cv2.imshow("Outline", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

check_if_cropped = False

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		check_if_cropped = True
		if len(refPt) == 2:
			roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			cv2.imshow("ROI", roi)
			y_end_cr = roi.shape[0]
			x_end_cr = roi.shape[1]
			cv2.waitKey(0)
		break
	elif key == ord("s"):
		roi = clone.copy()
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it

# close all open windows
cv2.destroyAllWindows()



if check_if_cropped == True :
	screenCnt = numpy.array([[[0,0]] ,[[x_end_cr,0]] ,[[x_end_cr, y_end_cr]] ,[[0 ,y_end_cr]]])


# apply the four point transform to obtain a top-downs
# view of the original image

#warped = four_point_transform(imutils.resize(roi, height = 500, width = 500), screenCnt.reshape(4, 2))
warped = four_point_transform(roi, screenCnt.reshape(4, 2)) 

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect


    # Threshold the HSV image to get only blue colors
warped = cv2.cvtColor(warped,cv2.COLOR_RGBA2GRAY)

T = threshold_local(warped, 11, offset = 5, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 500))
#cv2.imshow("Scanned", imutils.resize(warped, height = 500, width = 500))
cv2.imshow("Scanned", warped)

cv2.waitKey(0)