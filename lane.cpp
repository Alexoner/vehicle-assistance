/*************************************************************************
    > File Name: lane.cpp
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Thu 07 Mar 2013 02:44:22 PM CST
 ************************************************************************/

#include <iostream>
#include <algorithm>
#include <cv.h>
#include "utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <stdlib.h>
#include <stdio.h>

class Vehicle
{
    CvPoint bmin,bmax;
    int symmetryX;
    bool valid;
    unsigned int lastUpdata;
};


enum
{
    LINE_REJECT_DEGREES=60, //in degrees

    CANNY_MIN_THRESHOLD=1,//edge detector mininum hysteresis threshold
    CANNY_MAX_THRESHOLD=100,//edge detector maximum hysteresis threshold

    HOUGH_THRESHOLD=30,     //line approval vote threshold
    HOUGH_MIN_LINE_LENGTH=50, //remove lines shorter than this threshold
    HOUGH_MAX_LINE_GAP=100, //join lines

    LINE_LENGTH_DIFF=10,   //accepted diffenrence of length of lines,

    LANE_DYNAMPARAMS=2,//lane state vector dimension
    LANE_MEASUREPARAMS=2,//lane state vector dimension
    LANE_CONTROLPARAMS=0,//lane state vector dimension

    VEHICLE_DYNAMPARAMS=2,//vehicle state vector dimension
    VEHICLE_MEASUREPARAMS=2,//vehicle measurement dimension
    VEHICLE_CONTROLPARAMS=0,//vehicle control vector

    MAX_LOST_FRAME=30//maximum number of lost frames
};

int nframe=0;

bool sort_line_length(Line l0,Line l1)
{
    return l0.length < l1.length;
}

struct sort_line
{//near vertical line
    bool operator()(Line l0,Line l1)
    {
        return (l0.length*l0.length/fabsf(l0.angle-CV_PI/2) > l1.length*l1.length/fabsf(l1.angle-CV_PI/2));
    }
}sort_line_object;


cv::Mat frame,vmeandist,gray,blur,thrs,dil,ero,canny,dst;


int VMeanDist(cv::Mat src,cv::Mat &dist)
{
    cv::Size srcSize=src.size();
    int rows=srcSize.height,cols=srcSize.width;
    cv::Mat row(1,cols,CV_8UC1);//store a row
    dist=cv::Mat(1,rows,CV_8UC1);//the target distribution matrix
    int i,j,mean;
    //std::cout<<src.size()<<"size.height "<<srcSize.height<<std::endl;
    for(i=0;i<rows;i++)
    {//average row by row
        row=src.row(i);
        mean=0;
        for(j=0;j<row.cols;j++)
        {
            mean+=row.data[j];
        }
        mean/=row.cols;
        //std::cout<<"\nmean= "<<mean<<std::endl;
        dist.data[i]=mean;
    }
    /*for(i=0;i<dist.cols;i++)
    {
        std::cout<<i<<","<<(int)dist.data[i]<<";";
    }*/

    return 0;
}

int findHorizon(cv::Mat dist)
{
    int key=dist.data[0],i,lsum;//lsum is the local sum.
    for(i=1;i<dist.cols;i++)
    {
        if(dist.data[i]>key && i>1 && key<=10)
            return i;
        key=dist.data[i];
    }
    return i;
}

void filterLines(std::vector<Line> &lines,bool right)
{
    std::vector<int> eraselist;
    int i,a=-2*right+1;
    float angle,anglediff;
    if(!lines.size())
        return;
    for(i=0;i<(signed)lines.size();i++)
    {
        angle=lines[i].angle,anglediff=angle-M_PI_2;
        //reject lines of wrong slope angle
        if(anglediff*a<0||fabsf(anglediff)*180/CV_PI > LINE_REJECT_DEGREES)
        {
            lines.erase(lines.begin()+i);
            i--;
            //std::cout<<"near horizon"<<std::endl;
            continue;
        }
    }
    sort(lines.begin(),lines.end(),sort_line_object);
    //std::cout<<"size: "<<lines.size()<<std::endl;
    //sort the lines by degrees near vertical line
/*    if(lines.size())
    {
        sort(lines.begin(),lines.end(),sort_line_object);
        angle=lines[0].angle,lengc,th=lines[0].length;
        for(i=1;i<lines.size();i++)
        {
            if(fabsf(angle-lines[i].angle) < LINE_ANGLE_DIFF
               && fabsf(length-lines[i].length) > LINE_LENGTH_DIFF)
            {
                length=lines[i].length;
                angle=lines[i].angle;
            }
        }
*/
}

int processlines(std::vector<cv::Vec4i> lines,
                 cv::InputArray _edges,
                 cv::OutputArray _dst)
{
    std::vector<Line>left,right;
    cv::Mat dst=_dst.getMat();
    unsigned int i;

    for(i=0;i<lines.size();i++)
    {
        cv::Vec4i l=lines[i];
        CvPoint p0=cvPoint(l[0],l[1]),p1=cvPoint(l[2],l[3]);
        //assuming that the vanishing point is close to the image horizontal
        //center,calculate line parameters in form:y = kx + b;
        //decide line's side based on its midpoint position
        int midx=(l[0]+l[2])/2;
        if(midx<dst.cols/2)
        {
            left.push_back(Line(cvPoint(l[0],l[1]), cvPoint(l[2],l[3])));
        }
        else if(midx>dst.cols/2)
        {
            right.push_back(Line(cvPoint(l[0],l[1]), cvPoint(l[2],l[3])));
        }
    }

    for(int i=0;i<(int)left.size();i++)
    {
        cv::line(dst,left[i].p0,left[i].p1,CV_RGB(255,0,0),1);
        char str[20];
        sprintf(str,"%dth line,%f",i,calLength(left[i].p0,left[i].p1));
        cv::putText(dst,str,left[i].p0,1,1,1,1);
    }
    for(int i=0;i<(int)right.size();i++)
    {
        char str[20];
        sprintf(str,"%dth line,%f",i,right[i].angle);
        cv::putText(dst,str,right[i].p0,1,1,1,1);
        cv::line(dst,right[i].p0,right[i].p1,CV_RGB(0,255,0),1);
        if(i==right.size()-1)
        {
        //cv::line(dst,right[i].p0,right[i].p1,CV_RGB(23,21,10),5);
        }
    }

    filterLines(left,false);
    filterLines(right,true);
    if(left.size() && right.size())
    {
        cv::line(dst,left[left.size()-1].p0,left[left.size()-1].p1,CV_RGB(23,21,10),5);
        cv::line(dst,right[right.size()-1].p0,right[right.size()-1].p1,CV_RGB(23,21,10),5);
    }

    cv::line(dst,cvPoint(dst.cols/2,0),cvPoint(dst.cols/2,dst.rows),CV_RGB(0,0,0),1);

    //draw selected lanes
    int x1=dst.cols * 0.55f;
    int x2=dst.cols;
    //cv::line(frame,cvPoint(x1,laneR.k.get()*x1+laneR.b.get()),
    //        cvPoint(x2,laneR.k.get()*x2+laneR.b.get()),CV_RGB(255,0,255),3);

    return 0;
}

int wait(int k,int delay)
{
    if (k ==cv::waitKey(delay))
        return 1;
    return 0;
}

int detectLane(cv::Mat &frame,std::vector<cv::Vec4i>lines)
{
    int element_shape=cv::MORPH_RECT,an=1;
    int thrs1=0,thrs2=4000;
    double rho=1,theta=CV_PI/180;
    cv::Scalar color;
    int y,i,j;
    //cv::vector <cv::Vec2f> lines;
    //cv::vector <cv::Vec4i> lines;
    VMeanDist(frame,vmeandist);
    //std::cout<<vmeandist.data<<std::endl;
    y=findHorizon(vmeandist);
    for(i=0;i<y;i++)
    {
        for(j=0;j<canny.cols;j++)
        {
            frame.data[ero.cols*i+j]=0;
        }
    }

    cv::Canny(frame,canny,CANNY_MIN_THRESHOLD,CANNY_MAX_THRESHOLD,3);
    cv::imshow("canny",canny);

    cv::HoughLinesP(canny,lines,rho,theta,
                    HOUGH_THRESHOLD,HOUGH_MIN_LINE_LENGTH,HOUGH_MAX_LINE_GAP);
    //cv::HoughLines(canny,lines,rho,theta,HOUGH_THRESHOLD,3,8);
    processlines(lines,canny,dst);
    /*for(i=0;i<lines.size();i++)
    {
        cv::line(dst,cvPoint(lines[i][0],lines[i][1]),
                 cvPoint(lines[i][2],lines[i][3]),cv::Scalar(0,255,0),4,8);
    }*/
    //std::cout<<"\n"<<"y: "<<y<<std::endl;
    cv::line(dst,cvPoint(0,y),cvPoint(750,y),cv::Scalar(0,0,0),1,8,0);
}

int detectvehicle(cv::Mat &frame,std::vector<cv::Rect> &rects,
                  std::string cascade_name)
{//return the number of detected vehicles
    cv::CascadeClassifier vehicle(cascade_name);
    if(vehicle.empty())
    {
        std::cout<<"unable to load the classifier"<<std::endl;
        return -1;
    }
    vehicle.detectMultiScale(frame,rects,1.1, 2,0,cv::Size(80,80));
    cv::Rect p;
    //for(int i=0;i<rects.size();i++)
    /*for(int i=0;i<rects.size()&&i<1;i++)
    {
        p=rects[i];
        cv::rectangle(frame,cvPoint(p.x,p.y),cvPoint(p.x+p.width,p.y+p.height),
                      cv::Scalar(0,255,0),1,8,0);
    }*/
    return rects.size();
}

int trackline(cv::Mat& frame,
              std::vector<cv::Vec4i>& lines)
{
    static cv::KalmanFilter linekf;
    static cv::Mat x_k(LANE_DYNAMPARAMS,1,CV_32F);
    static cv::Mat w_k(LANE_DYNAMPARAMS,1,CV_32F);
    static cv::Mat z_k=cv::Mat::zeros(LANE_MEASUREPARAMS,1,CV_32F);
    static cv::Mat y_k=cv::Mat(LANE_DYNAMPARAMS,1,CV_32F);

    cv::Vec4i l;
    float k,b;

    int found=0;

    if(lines.size()==0)
    {//empty,then measure
        linekf.transitionMatrix=*(cv::Mat_<float>(2,2)<<1,0,0,1);
        setIdentity(linekf.measurementMatrix);
        setIdentity(linekf.processNoiseCov,cv::Scalar::all(1e-5));
        setIdentity(linekf.measurementNoiseCov,cv::Scalar::all(1e-1));
        setIdentity(linekf.errorCovPost,cv::Scalar::all(1));

        found=detectLane(frame,lines);//detect in the whole image scope
    }
    else
    {//predict and measure
        y_k=linekf.predict();//predict
        int dk=3*sqrt(linekf.errorCovPre.at<float>(0,0));//error band
        int db=3*sqrt(linekf.errorCovPre.at<float>(1,1));
        int top=frame.rows*2/3,bottom=frame.rows;
        Line l0(y_k.at<float>(0,0),y_k.at<float>(1,0)-db);
        Line l1(y_k.at<float>(0,0),y_k.at<float>(1,0)+db);
        CvPoint p0=cvPoint(l0.getx(bottom),bottom);
        CvPoint p1=cvPoint(l1.getx(top),top);
        cv::Rect roi(p0,p1);
        cv::Mat roiimage=frame(roi);//detect in the roi

        found=detectLane(roiimage,lines);
    }
    if(!found)
    {
        //lost++;
    }
    else
    {
        l=lines[0];
        k=calSlope(cvPoint(l[0],l[1]),cvPoint(l[2],l[3]));
        b=calIntercept(cvPoint(l[0],l[1]),cvPoint(l[2],l[3]));
        z_k=*(cv::Mat_<int>(2,1)<<k,b);
        //randn(w_k,cv::Scalar(0),
        //      cv::Scalar:all(linekf.processNoiseCov.at<float>(0,0)));
        linekf.correct(z_k);
    }
}

int trackvehicle(cv::Mat &frame,
                 std::vector<cv::Rect> &rects,
                 std::string cascade_name)
{//track a vehicle only
    //declare kalman filter object and related matrixes;
    static cv::KalmanFilter vehiclekf(VEHICLE_DYNAMPARAMS,
                                      VEHICLE_MEASUREPARAMS,
                                      VEHICLE_CONTROLPARAMS);
    static cv::Mat x_k(VEHICLE_DYNAMPARAMS,1,CV_32F);//state vector
    static cv::Mat w_k(VEHICLE_DYNAMPARAMS,1,CV_32F);
    static cv::Mat z_k=cv::Mat::zeros(VEHICLE_MEASUREPARAMS,1,CV_32F);
    //static cv::Mat y_k;
    //state matrix is [x,y] column vector
    static struct State
    {
        int lostframe;
    }state={0};
    int found=0;
    if(rects.size()==0)
    {//measure
        vehiclekf.transitionMatrix=*(cv::Mat_<float>(2,2)<<1,0,0,1);
        setIdentity(vehiclekf.measurementMatrix);
        setIdentity(vehiclekf.processNoiseCov,cv::Scalar::all(1e-5));
        setIdentity(vehiclekf.measurementNoiseCov,cv::Scalar::all(1e-1));
        setIdentity(vehiclekf.errorCovPost,cv::Scalar::all(1));
        found=detectvehicle(frame,rects,cascade_name);
        if(found<=0)
        {//no vehicle detected yet
            return 0;
        }
        else
        {
            //vehiclekf.statePost=*(cv::Mat_<int>(2,1)
            //                      << rects[0].x+rects[0].width/2,rects[0].y+rects[0].height/2);
            /*z_k=*(cv::Mat_<int>(2,1)<<rects[0].x+rects[0].width/2,rects[0].y+rects[0].height/2);
            vehiclekf.correct(z_k);*/
            cv::rectangle(frame,rects[0],cv::Scalar(0,255,255),1,8,0);
        }
    }
    else
    {//predict and measure
        cv::Mat y_k=vehiclekf.predict();
        //generate a region with predicted result,and detect in this area
        std::cout<<"372"<<std::endl;

        int dx=3*sqrt(vehiclekf.errorCovPre.at<float>(0,0));
        int dy=3*sqrt(vehiclekf.errorCovPre.at<float>(1,1));
        cv::Rect roi(y_k.at<int>(0,0)-dx-rects[0].width/2,
                     y_k.at<int>(1,0)-dy-rects[0].height/2,
                     rects[0].width+2*dx,rects[0].height+2*dy);
        //...

        //check roi effectiveness
        if(!(roi.x>=0&&roi.x+roi.width<=frame.cols&&
             roi.y>=0&&roi.y+roi.height<=frame.rows))
        {
            cv::putText(dst,"invalid region of interest",cvPoint(10,10),1,1,1,1);
            return 0;
        }
        cv::Mat roiimage=frame(roi);
        found=detectvehicle(roiimage,rects,cascade_name);
        if(found<=0)
        {
            state.lostframe++;
            if(state.lostframe>=MAX_LOST_FRAME)
            {
            //detect within a whole image
                found=detectvehicle(frame,rects,cascade_name);
                if(found<=0)
                {//the vehicle is totally lost
                    //still not found
                    state.lostframe=0;
                    //erase this vehicle and reset related data
                    return 0;//tracking 0 vehicle
                }
            }
        }//if(found<=0)
        else
        {
            //generat measurement
            z_k=*(cv::Mat_<int>(2,1) << rects[0].x,rects[0].y);
            randn(w_k,cv::Scalar(0),
                     cv::Scalar::all(sqrt(vehiclekf.processNoiseCov.at<float>(0,0))));
            vehiclekf.correct(z_k);
            cv::rectangle(frame,rects[0],cv::Scalar(0,255,0),1,8,0);
        }
    }
}


int main(int argc,char **argv)
{
    cv::VideoCapture cap;

    if( argc == 1 || (argc==2 && strlen(argv[1])==1 && isdigit(argv[1][0])))
       cap.open("data/video/test11_divx6.1.1.avi");
    else if( argc >= 2)
    {
        cap.open(argv[1]);
        if( cap.isOpened())
            std::cout << "Video "<<argv[1]<<
                ": width="<<cap.get(CV_CAP_PROP_FRAME_WIDTH)<<
                ",height="<<cap.get(CV_CAP_PROP_FRAME_HEIGHT)<<
                ",nframes="<<cap.get(CV_CAP_PROP_FRAME_COUNT)<<std::endl;
        if( argc>2 && isdigit(argv[2][0]))
        {
            int pos;
            sscanf(argv[2],"%d",&pos);
            std::cout<<"Seeking to frame #"<<pos<<std::endl;
            cap.set(CV_CAP_PROP_POS_FRAMES,pos);
        }
    }

    if(!cap.isOpened())
    {
        std::cout<<"Could not initialize capturing...\n";
        return -1;
    }

    cv::namedWindow("frame");
    /*
     * cv::namedWindow("blur");
    cv::namedWindow("thrs");
    cv::namedWindow("dil");
    cv::namedWindow("ero");
    */
    cv::namedWindow("canny");
    cv::namedWindow("dst");
    std::vector<cv::Vec4i> lines;
    std::vector<cv::Rect> rects;

    int c,ksize=3;
    for(;;)
    {
        cap>>frame;
        if(frame.empty())
            break;

        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
        cv::blur(gray,blur,cv::Size(ksize,ksize));
        cv::threshold(blur,thrs,150,255,cv::THRESH_BINARY);
        cv::dilate(thrs,dil,NULL);
        cv::erode(dil,ero,NULL);
        dst=frame.clone();

        detectLane(ero,lines);
        //detectvehicle(dst,rects,"data/cars3.xml");
        trackvehicle(dst,rects,"data/cars3.xml");

        cv::imshow("frame",frame);
        //cv::imshow("ero",ero);
        cv::imshow("dst",dst);

        c=cv::waitKey(10);
        if(c=='q'||c=='Q'||(c&255)==27)
            break;
        if(c==' ')
        {
            c=cv::waitKey();
            while(c!=' ')
            {
                c=cv::waitKey();
                if(c==' ')
                    break;
            }
        }
    }
    return 0;
}

