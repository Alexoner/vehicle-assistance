#!/usr/bin/env python

import sys
import os
import getopt
import cv
import cv2
import math

trackbar_smooth_name='smooth'
trackbar_threshold_name='Threshold'
trackbar_edge_name='edge'
window_src='src'
window_smooth='smooth'
window_threshold='threshold'
window_edge='edge'
window_output='output'

def smooth_trackbar(ksize):
	smooth_img=cv.CreateImage(cv.GetSize(src),8,1)
	cv.Smooth(src,smooth_img,cv.CV_GAUSSIAN)
	cv.ShowImage(window_smooth,smooth_img)

def threshold_trackbar(threshold):
	cv.Threshold(src,threshold_img,threshold,255,cv2.THRESH_BINARY)
	cv.ShowImage(window_threshold,threshold_img)

def edge_trackbar(position):
	#cv.Smooth(threshold_img,edge_img,cv.CV_BLUR,3,3,0)
	cv.Smooth(src,edge_img,cv.CV_BLUR,3,3,0)
	#cv.Not(src,edge_img)

	#run the edge detector on threshold scale
	cv.Canny(src,edge_img,position,position*3,3)

	#show the image
	cv.ShowImage(window_edge,edge_img)


if __name__=="__main__":
	try:
		img_path=sys.argv[1]
	except IndexError:
		#print str(IndexError)
		img_path='data/img/road.png'

	#.gray image
	src=cv.LoadImage(img_path,cv.CV_LOAD_IMAGE_GRAYSCALE)

	#create windows
	cv.NamedWindow(window_src,cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow(window_output,cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow(window_threshold,cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow(window_edge,cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow(window_smooth,cv.CV_WINDOW_AUTOSIZE)

	dst=cv.CreateImage(cv.GetSize(src),8,1)

	#create images
	threshold_img=cv.CreateImage((src.width,src.height),8,1)
	edge_img=cv.CreateImage((src.width,src.height),8,1)
	#cv.CvtColor(src,threshold_img,cv.CV_BGR2GRAY)

	#show the original image
	cv.ShowImage('src',src)

	#.filter image
	cv.CreateTrackbar(trackbar_smooth_name,window_smooth,1,19,smooth_trackbar)
	smooth_trackbar(3)

	#.threshold
	cv.CreateTrackbar(trackbar_threshold_name,window_threshold,0,255,threshold_trackbar)
	threshold_trackbar(100)

	#.edge detect
	cv.CreateTrackbar(trackbar_edge_name,window_edge,0,255,edge_trackbar)
	edge_trackbar(100)

	cv.Threshold(src,dst,200,255,cv2.THRESH_BINARY)
	cv.ShowImage(window_output,dst)
	color_dst = cv.CreateImage(cv.GetSize(src), 8, 3)
	storage=cv.CreateMemStorage(0)
	cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)
	#lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, math.pi / 180, 100, 0, 0)
	lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 0.5, math.pi / 180, 50,50,10)
	#lines=[]


	for (rho, theta) in lines[:100]:
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = a * rho
		y0 = b * rho
		pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
		pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
		cv.Line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 3, 8)

	cv.ShowImage('image',color_dst)


	if cv2.waitKey(0) & 0xFF == 27:
		cv2.destroyAllWindows()
