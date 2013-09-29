#!/usr/bin/env python

import sys
import cv2
import math

win_src='src'
win_dst='dst'


def threshold_trackbar(thrs_img,thrs=200):
	if thrs != 200:
		thrs=cv2.getTrackbarPos('thrs','threshold')
	ret,thrs_img=cv2.threshold(dst,thrs,255,cv2.THRESH_BINARY)
	cv2.imshow('threshold',thrs_img)

def canny_trackbar(dil):
	thrs1=cv2.getTrackbarPos('thrs1','canny')
	thrs2=cv2.getTrackbarPos('thrs2','canny')
	edge=cv2.Canny(dil,thrs1,thrs2,apertureSize=5)
	cv2.imshow('canny',dil)
	#cv2.waitKey(0)
	edge_img=img.copy()
	edge_img /= 2
	edge_img[edge != 0] = (0,255,0)
	cv2.imshow('canny',edge_img)

def wait(k):
	if k == cv2.waitKey():
		sys.exit(0)


def nothing(*argv):
	pass

if __name__=='__main__':
	if len(sys.argv) == 1:
		cap = cv2.VideoCapture('data/video/test11_divx6.1.1.avi')
	else:
		cap = cv2.VideoCapture(sys.argv[1])
		if not cap.isOpend():
			cap.cv2.VideoCapture(int(sys.argv[1]))

	if not cap.isOpened():
		print 'Cannot initialize video capture'
		sys.exit(-1)
		#fn='data/img/road.png'

	while True:
		ret,frame = cap.read()
		#create window
		cv2.namedWindow(win_src,cv2.CV_WINDOW_AUTOSIZE)

		img=frame.copy()
		#gray scale
		dst=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		cv2.imshow(win_src,dst)
		cv2.waitKey(0)

		#crop

		#smooth
		dst=cv2.blur(dst,(3,3))

		#compress low and high brightness values

		#equalize histogram
		#dst=cv2.equalizeHist(dst)

		#threshold
		#cv2.namedWindow('threshold',cv2.CV_WINDOW_AUTOSIZE)
		#cv2.createTrackbar('thrs','threshold',0,255,threshold_trackbar)
		ret,thrs_dst=cv2.threshold(dst,200,255,cv2.THRESH_BINARY)
		#threshold_trackbar(thrs_dst,200)

		#closure:dilate and erode
		if False:
			dil=cv2.dilate(thrs_dst,None)
			#cv2.namedWindow('dilate & erode')
			#cv2.imshow('dilate & erode',dil)
			#cv2.waitKey(0)
			ero=cv2.erode(dil,None)
			#cv2.imshow('dilate & erode',ero)
			#cv2.waitKey(0)
		else:
			ero=cv2.erode(thrs_dst,None)
			dil=cv2.dilate(ero,None)

		#edge detect:Canny
		#cv2.namedWindow('canny')
		edge_img=dil.copy()
		#cv2.createTrackbar('thrs1','canny',2000,5000,canny_trackbar)
		#cv2.createTrackbar('thrs2','canny',4000,5000,canny_trackbar)
		thrs1=2000
		thrs2=4000
		edge=cv2.Canny(dil,thrs1,thrs2,apertureSize=5)
		cv2.imshow('canny',dil)
		wait(27)
		edge_img2=img.copy()
		edge_img2 /= 2
		edge_img2[edge != 0] = (0,255,0)
		cv2.imshow('canny',edge_img2)
		wait(27)
		#canny_trackbar(dil)
		#cv2.waitKey(0)

		#houghlines
		#cv2.namedWindow('line')
		#cv2.createTrackbar('rho','line',1,1000,nothing)
		#cv2.createTrackbar('theta','line',1,100,nothing)
		rho=1
		#theta=math.pi/180
		theta=1
		hlthrs=15
		lines=cv2.HoughLines(edge_img,rho,theta,hlthrs)
		#lines=cv2.HoughLines(dst,rho,theta,hlthrs)
		storage=cv2.cv.CreateMemStorage(0)
		#lines=cv2.cv.HoughLines2(edge_img,storage,cv2.cv.CV_HOUGH_STANDARD,1,math.pi/180,100,0,0)

		#while True:
		#draw lines
		for (rho,theta) in lines[0]:
			print lines[0],rho,theta
			a=math.cos(theta)
			b=math.sin(theta)
			x0=a*rho
			y0=b*rho
			pt1=(cv2.cv.Round(x0+1000*(-b)),cv2.cv.Round(y0+1000*a))
			pt2=(cv2.cv.Round(x0-1000*(-b)),cv2.cv.Round(y0-1000*a))
			cv2.line(img,pt1,pt2,cv2.cv.RGB(0,0,255),3)

		cv2.imshow('line',img)
		if False:
			while True:
				k=cv2.waitKey(0)
				if k == 27:
					break


	cv2.destroyAllWindows()
