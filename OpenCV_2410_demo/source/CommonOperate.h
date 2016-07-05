/*******************************
*°æ±¾¸üÐÂ£º2015¡¢4¡¢29
*******************************/
#include "User_System.h"

using namespace std;
using namespace cv;

#ifndef __COMMON_OPERATE_H
#define __COMMON_OPERATE_H

vector<string> getNameSets(string front_part, string back_part, int numSets);
int addImagegap(cv::Mat&image, int gapx, int gapy, cv::Mat &image_gap);
void Mat2Point(cv::Mat &H, cv::Point3d&points);
void Point2Mat(cv::Mat &H, cv::Point3d&points);
void Mat2Vec(cv::Mat &H, vector<double>&H_vec);
void Mat2Vec(cv::Mat &H, vector<float>&H_vec);
void transformPoint(cv::Mat &H, vector<cv::Point3f> &Inpointsets, vector<cv::Point3f> &Outpointsets, int gapx = 0, int gapy = 0);
void transformPoint(cv::Mat &H, vector<cv::Point2f> &Inpointsets, vector<cv::Point2f> &Outpointsets, int gapx = 0, int gapy = 0);
void myWarpShow(cv::Mat &Img, cv::Mat &H);
void myimshow(const cv::Mat &Img);
vector<string>  get_filelist(char *foldname);
void getFiles(string path, vector<string>& files);
void getDirectories(string path, vector<string>& files);
void initvectordim2(int rows, int cols, vector<vector<cv::Mat> > &ivec);
void initvectordim2_H(int rows, int cols, vector<vector<cv::Mat> > &ivec, cv::Mat H);
void initvectordim1_H(int rows, vector<cv::Mat>&ivec, cv::Mat H);
void initvectordim2_o(int rows, int cols, vector<vector<cv::Point2f> > &ivec, cv::Point2f o);
vector<cv::Mat> initvectordim1_H2(int rows, cv::Mat H);
string getFilenameFromePath(string FilePath, char file[]);
cv::Point2f matMyPointCVMat(const cv::Point2f &pt, const cv::Mat H);
int FindMax(vector<int> inputvec, int &index);
double Average(vector<int> inputvec);
int FindMax(vector<double> inputvec, int &index);
void swap(int array[], int i, int j);
void swap(vector<int> &ivec, int i, int j);
void SelectionSort(int array[], int n);
void SelectionSort(vector <int>&ivec, vector <int>&Sortindex);
int findmaxarea(vector<int> inputvec, int windowsize);
void swap(vector<double> &ivec, int i, int j);
void SelectionSort(vector <double>&ivec, vector <int>&Sortindex, int flag);
int findmaxarea(vector<double> inputvec, int windowsize);
void MyDrawPolygon(Mat &img, vector<cv::Point2f>points);
void getImageNon_ZeroMask(cv::Mat image, cv::Mat &mask);
void getImageNon_ZeroMask(cv::Mat image, cv::Mat &mask, int value);
bool isPointinLine(cv::Point2f pt, cv::Point2f V1, cv::Point2f V2, double threshold = 0);
void getMaskShapeImage(cv::Mat SrcIm, cv::Mat &DstIm, cv::Mat mask);
void drawfeatures(cv::Mat &image, vector<cv::Point>features);
bool isPointInVec(cv::Point2f pt, vector<cv::Point2f>vec);
void removeRepeatPts(vector<cv::Point2f>quaryPts, vector<cv::Point2f>&trainPts);
bool removeRepeatPtsInoneVec(vector<cv::Point2f>&quaryPts, vector<cv::Point2f>&trainPts);
void findcorners(cv::Mat image, int num_corners, vector<cv::Point>&corners);
Rect findRect(vector<cv::Point>corners);
cv::Mat circleMatrix(int radius);
void findboun_rect(cv::Mat mask, vector<cv::Point>&corners);
cv::Mat calcgradient(const cv::Mat& img);
cv::Mat getMaskcontour(cv::Mat mask, int width);
double addROIPix(cv::Mat image, cv::Mat mask);
void SubMatrix(cv::Mat matrix1, cv::Mat matrix2, cv::Mat& result);
#endif




/*frequent using code*/

// time calculate
/**************************************************************
double t=(double)cvGetTickCount();

my code write here!

t=((double)cvGetTickCount() - t)/(cvGetTickFrequency()*1000);
cout << "my code time: " << t <<"ms"<< endl;
**************************************************************/
