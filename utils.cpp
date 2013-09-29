/*************************************************************************
    > File Name: utils.cpp
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Sun 17 Mar 2013 09:53:47 PM CST
 ************************************************************************/

#include "utils.h"

Line::Line()
    :p0(cvPoint(0,0)),p1(cvPoint(0,0)),angle(0),k(0),b(0),length(0) {}

Line::Line(CvPoint p0,CvPoint p1)
  :p0(p0),p1(p1),votes(0),visited(0),found(0)
{
  angle=calSlopeAngle(p0,p1);
  k=calSlope(p0,p1);
  b=p0.y-k*p0.x;
  length=calLength(p0,p1);
}

Line::Line(float k,float b)
  :p0(cvPoint(0,0)),p1(cvPoint(0,0)),angle(atan(k)),k(k),b(b),
  length(0),votes(0),visited(false), found(false){}

double Line::getx(float y)
{
    float x=(y-this->b)/this->k;
    return x;
}

double Line::gety(float x)
{
    float y=this->k*x+this->b;
    return y;
}

Line::~Line(){}

double calSlope(CvPoint p0,CvPoint p1)
{
    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;
    double slope;

    if(dy==0)
        return 0;
    else if(dx==0)
        return 0;
    else
        slope=dy/(double)dx;

    return slope;
}

double calSlopeAngle(CvPoint p0,CvPoint p1)
{
    //return the angle of slope in radius
    double theta=0;

    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;

    if(dx==0)
    {
        theta=CV_PI/2;
    }
    else if(dy==0)
    {
        theta=0;
    }
    else
    {
        theta=atan(dy/(double)dx);
    }
    if(theta<0)
        theta=M_PI+theta;
    return theta;
}

double calIntercept(CvPoint p0,CvPoint p1)
{
    int k=calSlope(p0,p1);
    int b=p0.y-k*p0.x;
    return b;
}

double calLength(CvPoint p0,CvPoint p1)
{
    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;

    return sqrt(dx*dx+dy*dy);
}

CvPoint midPoint(CvPoint p0,CvPoint p1)
{
    return cvPoint((p0.x+p1.x)/2,(p0.y+p1.y)/2);
}

CvPoint calIntersection(Line l0,Line l1)
{
    CvPoint intersection;
    intersection.x=(l1.b-l0.b)/(l0.k-l1.k);
    intersection.y=l0.k*intersection.x+l0.b;

    return intersection;
}

int pointIn(CvPoint pt,std::vector<CvPoint>& pts)
{
    //sort(pts,
    return 0;
}

