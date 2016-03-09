#include <opencv2\opencv.hpp>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std;




int VectorSave(vector<cv::Mat> &matrix, char*path);
int readPointSet_txt(string filename, vector<cv::Point2f>&SrcPoints, vector<cv::Point2f>&DetPoints);
int PointSave(vector<cv::Point2f> &SrcPoints, vector<cv::Point2f> &DstPoints, string path);
int readHlist_txt(char*filename, vector<cv::Mat>&HomoList, int numFrames);