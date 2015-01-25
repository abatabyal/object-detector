#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sstream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

Mat normalize(const Mat& src) {
    Mat srcnorm;
    normalize(src, srcnorm, 0, 255, NORM_MINMAX, CV_32FC1);
    return srcnorm;
}

const char* frame_window = "Frame";
const char* result_window = "Result window";

int main(int argc, char** argv)
{

float pi=3.14;
Mat f_gray, X, R, R1, R2, frame, framec, Y, Z, frame_display, vd, result, As, Bs, Cs, Ah, Bh, Ch;
Mat t1r, t2r, t3r, t4r, t5r, t6r, t7r, t8r, t9r, t10r;


VideoCapture capture(argv[1]);
if ( !capture.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

vector<Mat> vect;

Mat t1= imread("1.jpeg",0);
resize(t1, t1r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t1r);

Mat t2= imread("2.jpeg",0);
resize(t2, t2r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t2r);

Mat t3= imread("3.jpeg",0);
resize(t3, t3r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t3r);

Mat t4= imread("4.jpeg",0);
resize(t4, t4r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t4r);

Mat t5= imread("5.jpeg",0);
resize(t5, t5r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t5r);

Mat t6= imread("6.jpeg",0);
resize(t6, t6r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t6r);

Mat t7= imread("7.jpeg",0);
resize(t7, t7r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t7r);

Mat t8= imread("8.jpeg",0);
resize(t8, t8r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t8r);

Mat t9= imread("9.jpeg",0);
resize(t9, t9r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t9r);

Mat t10= imread("10.jpeg",0);
resize(t10, t10r, Size(90,90), 0, 0, CV_INTER_AREA ); 
vect.push_back( t10r);

int total = vect[0].rows * vect[0].cols;

Mat mat(total, vect.size(), CV_32FC1);
    for(int i = 0; i < vect.size(); i++) 
	{
        X = mat.col(i);
        vect[i].reshape(1, total).col(0).convertTo(X, CV_32FC1, 1/255.);
   	 }

int npc=3;

PCA pca(mat, Mat(), CV_PCA_DATA_AS_COL, npc);

Mat A = normalize(pca.eigenvectors.row(0).reshape(1, vect[0].rows));
double An = norm(A, NORM_L2);
Scalar Am = mean(A);
Mat Ap = (An * A);
add ( Ap, Am, As );
resize(As, Ah, Size(70,70), 0, 0, CV_INTER_AREA );

Mat B = normalize(pca.eigenvectors.row(1).reshape(1, vect[0].rows));
double Bn = norm(B, NORM_L2);
Scalar Bm = mean(B);
Mat Bp = (Bn * B);
add ( Bp, Bm, Bs );
resize(Bs, Bh, Size(70,70), 0, 0, CV_INTER_AREA );

Mat C = normalize(pca.eigenvectors.row(2).reshape(1, vect[0].rows));
double Cn = norm(C, NORM_L2);
Scalar Cm = mean(C);
Mat Cp = (Cn * C);
add ( Cp, Cm, Cs );
resize(Cs, Ch, Size(70,70), 0, 0, CV_INTER_AREA );

float D = pca.eigenvalues.at<float>(0,0);
float E = pca.eigenvalues.at<float>(0,1);
float F = pca.eigenvalues.at<float>(0,2);

ofstream of;
of.open ("Info.txt");

for(int i=0;;i++) //reading frames from the video
{

	capture >> frame;
	if(frame.empty())   
	break;


	cvtColor( frame, f_gray, CV_RGB2GRAY );
	f_gray.convertTo(framec, CV_32FC1);

	matchTemplate (framec, Ah, R, CV_TM_CCOEFF_NORMED);
	matchTemplate (framec, Bh, R1, CV_TM_CCOEFF_NORMED);
	matchTemplate (framec, Ch, R2, CV_TM_CCOEFF_NORMED);

	Mat e2, p1l;;
	Mat vs = (R.mul(R));
	exp (vs, vd);
	float e1 = (-1)/(2 * (D * D));
	pow (vd, e1, e2);
	Mat p1 = ( 1 / ( sqrt (2*pi) * sqrt (D))) * (e2);
        
	Mat e2a, vda, p1al;
	Mat vsa = (R1.mul(R1));
	exp (vsa, vda);
	float e1a = (-1)/(2 * (E * E));
	pow (vda, e1a, e2a);
	Mat p1a = ( 1 / ( sqrt (2*pi) * sqrt (D))) * (e2a);
	
	Mat e2b, vdb, p1bl;
	Mat vsb = (R2.mul(R2));
	exp (vsb, vdb);
	float e1b = (-1)/(2 * (F * F));
	pow (vdb, e1b, e2b);
	Mat p1b = ( 1 / ( sqrt (2*pi) * sqrt (D))) * (e2b);
	
	Mat p = (p1.mul(p1a)).mul(p1b);

	int result_cols =  frame.cols - p.cols + 1;
	int result_rows = frame.rows - p.rows + 1;

	result.create( result_cols, result_rows, CV_32FC1 );

	frame.copyTo( frame_display );

	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	matchLoc= maxLoc;

	/// Create windows
	namedWindow( frame_window, CV_WINDOW_AUTOSIZE );
	namedWindow( result_window, CV_WINDOW_AUTOSIZE );

	rectangle( frame_display, matchLoc, Point( matchLoc.x + p.cols , matchLoc.y + p.rows ), Scalar::all(0), 2, 8, 0 );
	rectangle( result, matchLoc, Point( matchLoc.x + p.cols , matchLoc.y + p.rows ), Scalar::all(0), 2, 8, 0 );

	of << "Canoe"<< "             " << i << "             " << minLoc<<"             "<< maxLoc << "\n";

	imshow( frame_window, frame_display );
	imshow( result_window, result );

	waitKey(0);

}

}


