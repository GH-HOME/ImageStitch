#include "User_System.h"
#include "opencv2/opencv_modules.hpp"
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
using namespace cv::detail;

#ifndef __MYIMAGESTITCH__
#define __MYIMAGESTITCH__

cv::Mat ImageStitch(vector<cv::Mat>images_source);
cv::Mat ImageStitch(vector<cv::Mat>image_warpeds, vector<cv::Mat>mask_warpeds, vector<cv::Mat>mask_graphcut, vector<cv::Point>corners);

#endif