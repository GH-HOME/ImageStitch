#include "CommonOperate.h"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;


#ifndef __IMAGESTITCH_H
#define __IMAGESTITCH_H


void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result, vector<cv::Mat> imagemask);
void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result);
void findcorner(cv::Mat image, vector<cv::Point2f>&corners, int num);
cv::Point2f findMiddlePoint(cv::Mat image);
void getTopLeftcorner(vector<cv::Point2f>corner, vector<cv::Point2f>&cornersrank);
void ImageStitch(vector<cv::Point> corners, vector<cv::Mat> images_warpngaps, vector<cv::Mat> mask_warpngaps,
	vector<cv::Point2f> maskcorner0, vector<cv::Point2f> maskcorner1, cv::Mat &result);
void getAppendixMask(cv::Mat mask, cv::Mat smallmask, vector<cv::Mat>&appendixmask, vector<cv::Point2f> maskvertex);
void rebuidmask(vector<cv::Mat>masktofindseam, vector<cv::Mat>appendixmask, vector<cv::Point2f>overlapvertex,
	vector<cv::Point2f>corners, vector<cv::Mat>&resultmask, cv::Mat overlapmask, vector<cv::Mat>mask_warpngaps);
int determineWhichMaskToUse(cv::Mat mask0, cv::Mat mask1, vector<cv::Point2f>overlapcorners);
void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result, vector<cv::Point>corners);
void ImageStitch(vector<cv::Mat> images_warpngaps, vector<cv::Mat> mask_warpngaps, cv::Mat& result);
#endif