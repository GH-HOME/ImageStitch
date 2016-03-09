#include "User_System.h"

using namespace std;
using namespace cv;

#define FEATURE_NUM_THRESHOLD 5

#ifndef __ROBUSTMATCHER_H
#define __ROBUSTMATCHER_H

class RobustMatcher
{
public:
	RobustMatcher();
	
private:
	cv::Ptr<cv::FeatureDetector>detector;
	cv::Ptr<cv::DescriptorExtractor>extractor;

	double m_ratio;
	double m_distance;
	double m_confidence;

public:
	cv::Mat m_Homography;
	std::vector<cv::Mat> m_FLists;
	vector<cv::KeyPoint> m_SourceFeature;
	vector<cv::KeyPoint> m_DestinateFeature;
	vector<vector<cv::KeyPoint>> m_SourceFeatureSets;
	vector<vector<cv::KeyPoint>> m_DestinateFeatureSets;
	vector<cv::Point2f>m_SourcePoint, m_DestPoint;
	vector<vector<cv::Point2f>>m_SourcePointSets, m_DestPointSets;
	std::vector<cv::DMatch> m_matches;

public:
	void setFeatureExtractor(cv::Ptr<cv::DescriptorExtractor>& desc);
	void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect);
	void setConfidenceLevel(double confid_level);
	void setMinDistanceToEpipolar(double distance);
	void setRatio(double ratio);
	cv::Mat match(cv::Mat &image1, cv::Mat &image2, std::vector<cv::DMatch>& matches);
	int matchList(vector<cv::Mat> &sourceImList, vector<cv::Mat> &DestImList,int flagfeatureout=0);
	int VideomatchList(vector<cv::Mat> &VideoFrameList, int flagfeatureout = 0);
	int ExtractFeatureSets();
	cv::Mat match_features(cv::Mat &image1, cv::Mat &image2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2);
	void match_features_noransac(cv::Mat &image1, cv::Mat &image2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2);
	int ClassClear();
private:
	void getfeaturePoint();
	int ratioTest(std::vector<std::vector<cv::DMatch>>&matches);
	void symmetryTest(
		const std::vector<std::vector<cv::DMatch>> &matches1,
		const std::vector<std::vector<cv::DMatch>> &matches2,
		std::vector<cv::DMatch>& symMatches);
	cv::Mat ransacTest(
		const std::vector<cv::DMatch>&matches,
		const std::vector<cv::KeyPoint> &keypoint1,
		const std::vector<cv::KeyPoint> &keypoint2,
		std::vector<cv::DMatch>&outMatches,int errflag);

};

void matches2points(const vector<KeyPoint>& keypoint1, const vector<KeyPoint>& keypoint2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& feature1,
	std::vector<Point2f>& feature2);

void EstimateFrameMotionFlannMatcher(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2, cv::Mat& H);
void EstimateFrameMotionForceMatcher(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2, cv::Mat& H);
#endif

void MyKLT_ComputeFeaturePairs4OneFrame(const Mat SourceImg, const Mat TargetImg
	, vector<Point2f> &SourceFeatures, vector<Point2f> &TargetFeatures);