#include "RobustMatcher.h"

using namespace std;
using namespace cv;


RobustMatcher::RobustMatcher() :m_ratio(0.7f), m_confidence(0.99), m_distance(7.0)
{
	detector = new cv::SiftFeatureDetector();
	extractor = new cv::SiftDescriptorExtractor();
}


void RobustMatcher::setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect)
{
	this->detector = detect;
}

void RobustMatcher::setFeatureExtractor(cv::Ptr<cv::DescriptorExtractor>& desc)
{
	this->extractor = desc;
}


void RobustMatcher::setConfidenceLevel(double confidence)
{
	this->m_confidence = confidence;
}

void RobustMatcher::setMinDistanceToEpipolar(double distance)
{
	this->m_distance = distance;
}

void RobustMatcher::setRatio(double ratio)
{
	this->m_ratio = ratio;
}
cv::Mat RobustMatcher::match(cv::Mat &image1,cv::Mat &image2,std::vector<cv::DMatch>& matches)
{ 
	m_SourceFeature.clear();
	m_DestinateFeature.clear();
	detector->detect(image1, m_SourceFeature);
	detector->detect(image2, m_DestinateFeature);

	cv::Mat descriptor1, descriptor2;
	extractor->compute(image1, m_SourceFeature, descriptor1);
	extractor->compute(image2, m_DestinateFeature, descriptor2);

	cv::BruteForceMatcher<cv::L2<float>>matcher;
	cv::vector<std::vector<cv::DMatch>>matches1;
	matcher.knnMatch(descriptor1, descriptor2, matches1, 2);

	cv::vector<std::vector<cv::DMatch>>matches2;
	matcher.knnMatch(descriptor2, descriptor1, matches2, 2);

	int remove = ratioTest(matches1);
	remove = ratioTest(matches2);
	std::vector<cv::DMatch>symMatches;
	symmetryTest(matches1, matches2, symMatches);
	int errflag = 0;
	cv::Mat fundemental = ransacTest(symMatches, m_SourceFeature, m_DestinateFeature, matches,errflag);
	this->m_matches = matches;
	this->m_Homography = fundemental;
		return fundemental;


}


int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch>>&matches)
{
	int removed = 0;
	for (std::vector<std::vector<cv::DMatch>>::iterator
		matchIterator = matches.begin();
		matchIterator != matches.end();++matchIterator)
	{
		if (matchIterator->size() > 1)
		{
			if ((*matchIterator)[0].distance /
				(*matchIterator)[1].distance > m_ratio)
			{
				matchIterator->clear();
				removed++;
			}
		}
		else
		{
			matchIterator->clear();
			removed++;
		}
	}
	return removed;
}

void RobustMatcher::symmetryTest(
	const std::vector<std::vector<cv::DMatch>> &matches1,
	const std::vector<std::vector<cv::DMatch>> &matches2,
	std::vector<cv::DMatch>& symMatches)
{
	for (std::vector<std::vector<cv::DMatch>>::const_iterator
		matchIterator1 = matches1.begin();
		matchIterator1 != matches1.end(); ++matchIterator1)
	{
		if (matchIterator1->size() <2)
			continue;
		for (std::vector<std::vector<cv::DMatch>>::const_iterator
			matchIterator2 = matches2.begin();
			matchIterator2 != matches2.end(); ++matchIterator2)
		{
			if (matchIterator2->size() < 2)
				continue;

			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
			{
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
				break;
			}
		}
	}
}

cv::Mat RobustMatcher::ransacTest(
	const std::vector<cv::DMatch>&matches,
	const std::vector<cv::KeyPoint> &keypoint1,
	const std::vector<cv::KeyPoint> &keypoint2,
	std::vector<cv::DMatch>&outMatches,int errflag)
{
	std::vector<cv::Point2f>points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		float x = keypoint1[it->queryIdx].pt.x;
		float y = keypoint1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));

		x = keypoint2[it->trainIdx].pt.x;
		y = keypoint2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}

	if (points1.size()<FEATURE_NUM_THRESHOLD || points2.size()<FEATURE_NUM_THRESHOLD)
	{
		errflag = 1;
		cerr << "num of feature points less than FEATURE_NUM_THRESHOLD: "<<FEATURE_NUM_THRESHOLD<< endl;
		return cv::Mat::eye(3, 3, CV_64F);
	}
	std::vector<uchar>inliers(points1.size(), 0);

	cv::Mat fundemental;
	fundemental = cv::findHomography(cv::Mat(points1),
		cv::Mat(points2),
		inliers,
		CV_LMEDS,
		m_distance);


	std::vector<uchar>::const_iterator itIn = inliers.begin();
	std::vector<cv::DMatch>::const_iterator itM = matches.begin();

	for (; itIn != inliers.end(); ++itIn,++itM)
	{
		if (*itIn)
		{
			outMatches.push_back(*itM);
		}
	}
	return fundemental;


}


int RobustMatcher::matchList(vector<cv::Mat> &sourceImList, vector<cv::Mat> &DestImList, int flagfeatureout)
{
	int size = sourceImList.size();
	std::vector<cv::DMatch>matches;
	printf("Image Matching...");
	for (int i = 0; i < size;i++)
	{
		cv::Mat Homography = match(sourceImList[i], DestImList[i],matches);
		printf("%04d\b\b\b\b", i);
		m_FLists.push_back(Homography);
		if (flagfeatureout)
		{
			ExtractFeatureSets();
		}
	}

	return 1;
}

int RobustMatcher::VideomatchList(vector<cv::Mat> &VideoFrameList,int flagfeatureout)
{
	int size = VideoFrameList.size();
	std::vector<cv::DMatch>matches;
	printf("Video Frame Matching...");
	for (int i = 0; i < size-1; i++)
	{
		cv::Mat homography = match(VideoFrameList[i], VideoFrameList[i+1], matches);
		m_FLists.push_back(homography);
		printf("%04d\b\b\b\b", i);
		if (flagfeatureout)
		{
			ExtractFeatureSets();
		}
	}
	
	return 1;
}

int RobustMatcher::ClassClear()
{
	m_FLists.clear();
	m_SourceFeatureSets.clear();
	m_DestinateFeatureSets.clear();
	m_Homography = cv::Mat::eye(3, 3, CV_64F);
	m_SourceFeature.clear();
	m_DestinateFeature.clear();
	return 1;
}

void matches2points(const vector<KeyPoint>& keypoint1, const vector<KeyPoint>& keypoint2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& feature1,
	std::vector<Point2f>& feature2)
{

	feature1.clear();
	feature2.clear();
	feature1.reserve(matches.size());
	feature2.reserve(matches.size());

	size_t i = 0;

	for (; i < matches.size(); i++)
	{

		const DMatch & dmatch = matches[i];

		feature2.push_back(keypoint2[dmatch.trainIdx].pt);
		feature1.push_back(keypoint1[dmatch.queryIdx].pt);

	}

}


int RobustMatcher::ExtractFeatureSets()
{
	
	for (std::vector<cv::DMatch>::const_iterator it = m_matches.begin(); it != m_matches.end(); ++it)
	{
		m_SourceFeatureSets.push_back(m_SourceFeature);
		m_DestinateFeatureSets.push_back(m_DestinateFeature);
		float x = m_SourceFeature[it->queryIdx].pt.x;
		float y = m_SourceFeature[it->queryIdx].pt.y;
		m_SourcePoint.push_back(cv::Point2f(x, y));

		x = m_DestinateFeature[it->trainIdx].pt.x;
		y = m_DestinateFeature[it->trainIdx].pt.y;
		m_DestPoint.push_back(cv::Point2f(x, y));
	}
	m_DestPointSets.push_back(m_DestPoint);
	m_SourcePointSets.push_back(m_SourcePoint);
	return 1;
}

void EstimateFrameMotionFlannMatcher(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2, cv::Mat& H)
{

	int minHessian = 400;
	cv::SurfFeatureDetector detector(minHessian);
	std::vector<cv::KeyPoint> keys1, keys2;
	detector.detect(Img1, keys1);
	detector.detect(Img2, keys2);
	cv::SurfDescriptorExtractor extractor;
	cv::Mat surfdescriptors1, surfdescriptors2;
	extractor.compute(Img1, keys1, surfdescriptors1);
	extractor.compute(Img2, keys2, surfdescriptors2);
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > surfmatches;
	matcher.match(surfdescriptors1, surfdescriptors2, surfmatches);
	double max = 0; double min = 100;
	for (int i = 0; i < surfdescriptors1.rows; i++)
	{
		double dist = surfmatches[i].distance;
		if (dist < min) min = dist;
		if (dist > max) max = dist;
	}

	for (int i = 0; i < surfdescriptors1.rows; i++)
	{
		if (surfmatches[i].distance < 0.5*max)
		{
			features1.push_back(keys1[surfmatches[i].queryIdx].pt);
			features2.push_back(keys2[surfmatches[i].trainIdx].pt);
		}
	}

	vector<uchar>inliers(features1.size(), 0);
	Mat H_prev = cv::findHomography(cv::Mat(features1),
		cv::Mat(features1),
		inliers,
		CV_RANSAC,
		1.);
	H_prev.copyTo(H);
}


cv::Mat RobustMatcher::match_features(cv::Mat &image1, cv::Mat &image2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2)
{
	std::vector<cv::DMatch>matches;
	m_SourceFeature.clear();
	m_DestinateFeature.clear();
	detector->detect(image1, m_SourceFeature);
	detector->detect(image2, m_DestinateFeature);

	cv::Mat descriptor1, descriptor2;
	extractor->compute(image1, m_SourceFeature, descriptor1);
	extractor->compute(image2, m_DestinateFeature, descriptor2);

	cv::BruteForceMatcher<cv::L2<float>>matcher;
	cv::vector<std::vector<cv::DMatch>>matches1;
	matcher.knnMatch(descriptor1, descriptor2, matches1, 2);

	cv::vector<std::vector<cv::DMatch>>matches2;
	matcher.knnMatch(descriptor2, descriptor1, matches2, 2);

	int remove = ratioTest(matches1);
	remove = ratioTest(matches2);
	std::vector<cv::DMatch>symMatches;
	symmetryTest(matches1, matches2, symMatches);
	int errflag = 0;
	cv::Mat fundemental = ransacTest(symMatches, m_SourceFeature, m_DestinateFeature, matches,errflag);
	if (errflag==0)
	{
		matches2points(m_SourceFeature, m_DestinateFeature, matches, features1, features2);
	}
	else
	{
		features1.resize(0);
		features2.resize(0);
	}
	
	return fundemental;


}

void RobustMatcher::match_features_noransac(cv::Mat &image1, cv::Mat &image2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2)
{
	std::vector<cv::DMatch>matches;
	m_SourceFeature.clear();
	m_DestinateFeature.clear();
	detector->detect(image1, m_SourceFeature);
	detector->detect(image2, m_DestinateFeature);

	cv::Mat descriptor1, descriptor2;
	extractor->compute(image1, m_SourceFeature, descriptor1);
	extractor->compute(image2, m_DestinateFeature, descriptor2);

	cv::BruteForceMatcher<cv::L2<float>>matcher;
	cv::vector<std::vector<cv::DMatch>>matches1;
	matcher.knnMatch(descriptor1, descriptor2, matches1, 2);

	cv::vector<std::vector<cv::DMatch>>matches2;
	matcher.knnMatch(descriptor2, descriptor1, matches2, 2);

	int remove = ratioTest(matches1);
	remove = ratioTest(matches2);
	std::vector<cv::DMatch>symMatches;
	symmetryTest(matches1, matches2, symMatches);

	matches2points(m_SourceFeature, m_DestinateFeature, symMatches, features1, features2);



}

void EstimateFrameMotionForceMatcher(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2, cv::Mat& H)
{
	cv::Mat I1_gray, I2_gray;
	double hessianThreshold(400.0);
	vector<cv::KeyPoint> keypoints1;
	vector<cv::KeyPoint> keypoints2;
	cv::SurfFeatureDetector SURF(hessianThreshold);
	cv::SurfDescriptorExtractor surfDesc;
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	cv::BruteForceMatcher<cv::L2<float>> matcher;
	
	cvtColor(Img1, I1_gray, CV_BGR2GRAY);
	cvtColor(Img2, I2_gray, CV_BGR2GRAY);

	//detect surf features
	SURF.detect(I1_gray, keypoints1);
	SURF.detect(I2_gray, keypoints2);

	//describe surf features
	surfDesc.compute(I1_gray, keypoints1, descriptors1);
	surfDesc.compute(I2_gray, keypoints2, descriptors2);

	std::vector<cv::DMatch> matches;

	matcher.match(descriptors1, descriptors2, matches);
	matches2points(keypoints1, keypoints2, matches, features1, features2);

	vector<uchar>inliers(features1.size(), 0);
	Mat H_prev = cv::findHomography(cv::Mat(features1),
		cv::Mat(features2),
		inliers,
		CV_RANSAC,
		1.);
	H_prev.copyTo(H);
}



void MyKLT_ComputeFeaturePairs4OneFrame(const Mat SourceImg, const Mat TargetImg
	, vector<Point2f> &SourceFeatures, vector<Point2f> &TargetFeatures)
{
	Mat img0Gray = Mat::zeros(SourceImg.rows, SourceImg.cols, CV_8UC1);
	Mat curImgGray = Mat::zeros(TargetImg.rows, TargetImg.cols, CV_8UC1);
	cvtColor(SourceImg, img0Gray, CV_RGB2GRAY);
	cvtColor(TargetImg, curImgGray, CV_RGB2GRAY);

	vector<Point2f> featurePtSet0;
	int maxNum = 10000;
	goodFeaturesToTrack(img0Gray, featurePtSet0, maxNum, 0.05, 5);
	cornerSubPix(img0Gray, featurePtSet0, Size(15, 15), Size(-1, -1)
		, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

	vector<Point2f> curfeaturePtSet;
	vector<uchar> curFlag;
	vector<float> curErrSet;
	calcOpticalFlowPyrLK(img0Gray, curImgGray, featurePtSet0, curfeaturePtSet, curFlag, curErrSet, Size(15, 15));
	for (int p = 0; p < curErrSet.size(); p++)
	{
		if (curErrSet[p] > 100 || curfeaturePtSet[p].x < 0 || curfeaturePtSet[p].y < 0
			|| curfeaturePtSet[p].x > img0Gray.cols || curfeaturePtSet[p].y > img0Gray.rows)
			curFlag[p] = 0;
	}
	for (int i = 0; i < curFlag.size(); i++){
		if (curFlag.at(i) == 1){
			SourceFeatures.push_back(featurePtSet0.at(i));
			TargetFeatures.push_back(curfeaturePtSet.at(i));
		}
	}
}


//vector<Point2f> SourceFeatures, TargetFeatures;
//MyKLT_ComputeFeaturePairs4OneFrame(InFrameList[i]
//	, InFrameList[i - 1], SourceFeatures, TargetFeatures);
//
//if (SourceFeatures.size() < 10) H = Mat::eye(3, 3, CV_64F);
//else	H = findHomography(Mat(SourceFeatures), Mat(TargetFeatures), CV_RANSAC);