/*************************************************************************
    > File Name: utils.h
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Sun 17 Mar 2013 10:56:35 PM CST
 ************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <cv.h>
class Line
{
 public:
  CvPoint p0,p1;
  float angle,k,b,length;
  int votes;
  bool visited,found;
 public:
  Line();
  Line(CvPoint p0,CvPoint p1);
  Line(float k,float b);
  virtual double getx(float y);
  virtual double gety(float x);
  bool operator <(const Line &l) const
  {
      return (this->length < l.length);
  }
  virtual ~Line();
};

//return slope of the line formed by two points
extern double calSlope(CvPoint p0,CvPoint p1);

//return the slope angle of the line formed by two points
extern double calSlopeAngle(CvPoint p0,CvPoint p1);

//return the intercept of line y=k*x+b
extern double calIntercept(CvPoint p0,CvPoint p1);

//return of the lenght of two points
extern double calLength(CvPoint p0,CvPoint p1);

//return the middle point
extern CvPoint midPoint(CvPoint p0,CvPoint p1);

//return intersection of two line
extern CvPoint calIntersection(Line l0,Line l1);

//check whether a point is in the *** formed by the pts
extern int pointIn(CvPoint pt,std::vector<CvPoint>& pts);

#endif
