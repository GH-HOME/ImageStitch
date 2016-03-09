#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stdio.h>
#include <vector>


using namespace std;
using namespace cv;

#ifndef __MYDEBUG__
#define __MYDEBUG__

void MyPrint(cv::Mat &H);
int MatrixShow(const cv::Mat &matrix, char* fileName);
int MatrixShow(vector<cv::Mat> &matrixList, char* fileName);
void MyPrint(vector<cv::Mat> &HList);

#endif
