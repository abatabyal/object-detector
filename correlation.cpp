#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace cv;
	
using namespace std;

Mat templ, templc, templr, templh, result, frame, framec, frameh, frame_display;

int main(int argc, char** argv)
{

const char* frame_window = "Frame";
const char* result_window = "Result window";

VideoCapture capture(argv[1]);

templ = imread(argv[2],1);
templ.convertTo(templc, CV_32FC1);
resize(templc, templh, Size(70,70), 0, 0, CV_INTER_AREA ); 



if ( !capture.isOpened() )  
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }
ofstream of;
of.open ("Info.txt");// opening the output text file

for(int i=0;;i++) //reading frames from the video
{
	capture >> frame;
	if(frame.empty())   
	break;
	frame.convertTo(frameh, CV_32FC1);
	
	int result_cols =  frameh.cols - templh.cols + 1;
	int result_rows = frameh.rows - templh.rows + 1;

	result.create( result_cols, result_rows, CV_32FC1 );

	frame.copyTo( frame_display );

	matchTemplate(frameh, templh, result, CV_TM_CCORR_NORMED);
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Create windows
	namedWindow( frame_window, CV_WINDOW_AUTOSIZE );
	namedWindow( result_window, CV_WINDOW_AUTOSIZE );
	
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	matchLoc= maxLoc;

	of << "Basketball"<< "  " << i << "   " <<"     "<< minLoc<< "   "<< maxLoc << "\n";

	rectangle( frame_display, matchLoc, Point( matchLoc.x + templh.cols , matchLoc.y + templh.rows ), Scalar::all(0), 2, 8, 0 );
	rectangle( result, matchLoc, Point( matchLoc.x + templh.cols , matchLoc.y + templh.rows ), Scalar::all(0), 2, 8, 0 );
	
	imshow( frame_window, frame_display );
	imshow( result_window, result );

	waitKey(0);
}
}


