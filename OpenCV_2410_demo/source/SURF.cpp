#include "SURF.h"
#include "Mesh.h"
#include "CommonOperate.h"

void mySURF(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2){

	int minHessian = 320;
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
		if (surfmatches[i].distance < 0.15*max)
		{
			features1.push_back(keys1[surfmatches[i].queryIdx].pt);
			features2.push_back(keys2[surfmatches[i].trainIdx].pt);
		}
	}
}

void mySift(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2){

	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keys1, keys2;
	detector.detect(Img1, keys1);
	detector.detect(Img2, keys2);
	cv::SiftDescriptorExtractor extractor;
	cv::Mat siftdescriptors1, siftdescriptors2;
	extractor.compute(Img1, keys1, siftdescriptors1);
	extractor.compute(Img2, keys2, siftdescriptors2);
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > siftmatches;
	matcher.match(siftdescriptors1, siftdescriptors2, siftmatches);
	//double max = 0; double min = 100;
	//for (int i = 0; i < siftdescriptors1.rows; i++)
	//{
	//	double dist = siftmatches[i].distance;
	//	if (dist < min) min = dist;
	//	if (dist > max) max = dist;
	//}

	for (int i = 0; i < siftdescriptors1.rows; i++)
	{
		features1.push_back(keys1[siftmatches[i].queryIdx].pt);
		features2.push_back(keys2[siftmatches[i].trainIdx].pt);
	}
}

void GlobalOutLinerRejectorOneIteration_SURF(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold){

	if (features_img1.size() > 10){
		vector<cv::Point2f> temp1, temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for (int i = 0; i < features_img1.size(); i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		vector<uchar> mask;
		cv::findHomography(cv::Mat(temp1), cv::Mat(temp2), mask, CV_RANSAC, threshold);

		for (int k = 0; k < mask.size(); k++){
			if (mask[k] == 1){
				features_img1.push_back(temp1[k]);
				features_img2.push_back(temp2[k]);
			}
		}
	}
	else{
		return;
	}
}



void ImageSizeRejectorOneIteration_SURF(int img_Height, int img_Width, vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2){

	vector<cv::Point2f> temp1, temp2;
	temp1.resize(features_img1.size());
	temp2.resize(features_img2.size());
	for (int i = 0; i < features_img1.size(); i++){
		temp1[i] = features_img1[i];
		temp2[i] = features_img2[i];
	}
	features_img1.clear();
	features_img2.clear();

	for (int k = 0; k < temp1.size(); k++){
		if (
			(temp1[k].x>0 && temp1[k].x <= img_Width - 1 &&
			temp1[k].y>0 && temp1[k].y <= img_Height - 1) &&
			(temp2[k].x > 0 && temp2[k].x <= img_Width - 1 &&
			temp2[k].y > 0 && temp2[k].y <= img_Height - 1))
		{
			features_img1.push_back(temp1[k]);
			features_img2.push_back(temp2[k]);
		}
	}
}




void MyRANSAC(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold,Size imageSize)
{
	vector<cv::Point2f> result1,result2;
	result1.resize(0);
	result2.resize(0);
	if (features_img1.size() > 10){
		vector<cv::Point2f> temp1, temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for (int i = 0; i < features_img1.size(); i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		int width = 320;
		int height = 160;
		for (int i = 0; i < imageSize.height;i+=10)
		{
			for (int j = 0; j < imageSize.width; j+=10)
			{
				vector<cv::Point2f> subtemp1, subtemp2;
				subtemp1.resize(features_img1.size());
				subtemp2.resize(features_img2.size());
				subtemp1 = temp1;
				subtemp2 = temp2;
				Point2f tl(j, i);
				//cout << tl << endl;
				Quad Subimage(tl, width, height);
				bool isreturn=QuadRansac(subtemp1, subtemp2, Subimage, threshold);
				if (isreturn)
				{
					removeRepeatPts(result1, subtemp1);
					removeRepeatPts(result2, subtemp2);
					result1.insert(result1.end(), subtemp1.begin(), subtemp1.end());
					result2.insert(result2.end(), subtemp2.begin(), subtemp2.end());
					if (result1.size() != result2.size())
					{
						cout << error << endl;
					}
				}
				
			}
		}
		features_img1 = result1;
		features_img2 = result2;

	}
}
	

//bool QuadRansac(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, Quad Imagequad,double threshold,cv::Mat src,cv::Mat dst)
//{
//	vector<cv::Point2f>quadfeatures1, quadfeatures2;
//	vector<int>pointIndex;
//	pointIndex.resize(0);
//	quadfeatures1.resize(0);
//	quadfeatures2.resize(0);
//	
//	for (int k = 0; k < features_img1.size(); k++)
//	{
//		cv::Point2f pt;
//		pt.x = features_img1.at(k).x;
//		pt.y = features_img1.at(k).y;
//		if (Imagequad.isPointIn(pt))
//		{
//			quadfeatures1.push_back(pt);
//			pointIndex.push_back(k);
//		}
//	}
//	for (int m = 0; m < pointIndex.size();m++)
//	{
//		quadfeatures2.push_back(features_img2[pointIndex[m]]);
//	}
//
//	cv::Mat srcyemp = src.clone();
//	cv::Mat dstyemp = dst.clone();
//	/*drawfeatures(srcyemp, quadfeatures1);
//	drawfeatures(dstyemp, quadfeatures2);
//	imshow("source", srcyemp);
//	imshow("target", dstyemp);
//	waitKey(20);*/
//
//
//	if (quadfeatures1.size()<4)
//	{
//		return false;
//	}
//	features_img1.resize(0);
//	features_img2.resize(0);
//	cout << quadfeatures1.size() << endl;
//	cout << quadfeatures2.size() << endl;
//	vector<uchar> mask;
//	findHomography(cv::Mat(quadfeatures1), cv::Mat(quadfeatures2), mask, CV_RANSAC, threshold);
//	for (int k = 0; k < mask.size(); k++){
//		if (mask[k] == 1){
//			features_img1.push_back(quadfeatures1[k]);
//			features_img2.push_back(quadfeatures2[k]);
//		}
//	}
//	cout << features_img1.size() << endl;
//	cout << features_img2.size() << endl;
//
//	cv::Mat srcyemp1 = src.clone();
//	cv::Mat dstyemp1 = dst.clone();
//	drawfeatures(srcyemp1, features_img1);
//	drawfeatures(dstyemp1, features_img2);
//	imshow("source1", srcyemp1);
//	imshow("target1", dstyemp1);
//	waitKey(20);
//
//	return true;
//
//}


bool QuadRansac(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, Quad Imagequad, double threshold)
{
	vector<cv::Point2f>quadfeatures1, quadfeatures2;
	vector<int>pointIndex;
	pointIndex.resize(0);
	quadfeatures1.resize(0);
	quadfeatures2.resize(0);

	for (int k = 0; k < features_img1.size(); k++)
	{
		cv::Point2f pt;
		pt.x = features_img1.at(k).x;
		pt.y = features_img1.at(k).y;
		if (Imagequad.isPointIn(pt))
		{
			quadfeatures1.push_back(pt);
			pointIndex.push_back(k);
		}
	}
	for (int m = 0; m < pointIndex.size(); m++)
	{
		quadfeatures2.push_back(features_img2[pointIndex[m]]);
	}

	
	

	if ((quadfeatures1.size() < 4) || (quadfeatures2.size() < 4))
	{
		return false;
	}
	if (quadfeatures1.size() != quadfeatures2.size())
	{
		return false;
	}
	features_img1.resize(0);
	features_img2.resize(0);
	

	vector<uchar> mask;
	findHomography(cv::Mat(quadfeatures1), cv::Mat(quadfeatures2), mask, CV_RANSAC, threshold);
	for (int k = 0; k < mask.size(); k++){
		if (mask[k] == 1){
			features_img1.push_back(quadfeatures1[k]);
			features_img2.push_back(quadfeatures2[k]);
		}
	}
	

	

	return true;

}


void MyRANSAC(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold, Size imageSize, cv::Mat src, cv::Mat dst)
{
	vector<cv::Point2f> result1, result2;
	result1.resize(0);
	result2.resize(0);
	if (features_img1.size() > 10){
		vector<cv::Point2f> temp1, temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for (int i = 0; i < features_img1.size(); i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		int width = 320;
		int height = 160;
		for (int i = 0; i < imageSize.height; i += 10)
		{
			for (int j = 0; j < imageSize.width; j += 10)
			{
				vector<cv::Point2f> subtemp1, subtemp2;
				subtemp1.resize(features_img1.size());
				subtemp2.resize(features_img2.size());
				subtemp1 = temp1;
				subtemp2 = temp2;
				Point2f tl(j, i);
				//cout << tl << endl;
				Quad Subimage(tl, width, height);
				bool isreturn = QuadRansac(subtemp1, subtemp2, Subimage, threshold, src, dst);
				if (isreturn)
				{
					removeRepeatPts(result1, subtemp1);
					removeRepeatPts(result2, subtemp2);
					result1.insert(result1.end(), subtemp1.begin(), subtemp1.end());
					result2.insert(result2.end(), subtemp2.begin(), subtemp2.end());
					if (result1.size() != result2.size())
					{
						cout << error << endl;
					}
				}

			}
		}
		features_img1 = result1;
		features_img2 = result2;

	}
}


bool QuadRansac(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, Quad Imagequad, double threshold, cv::Mat src, cv::Mat dst)
{
	vector<cv::Point2f>quadfeatures1, quadfeatures2;
	vector<int>pointIndex;
	pointIndex.resize(0);
	quadfeatures1.resize(0);
	quadfeatures2.resize(0);

	for (int k = 0; k < features_img1.size(); k++)
	{
		cv::Point2f pt;
		pt.x = features_img1.at(k).x;
		pt.y = features_img1.at(k).y;
		if (Imagequad.isPointIn(pt))
		{
			quadfeatures1.push_back(pt);
			pointIndex.push_back(k);
		}
	}
	for (int m = 0; m < pointIndex.size(); m++)
	{
		quadfeatures2.push_back(features_img2[pointIndex[m]]);
	}

	cv::Mat srcyemp = src.clone();
	cv::Mat dstyemp = dst.clone();
	/*drawfeatures(srcyemp, quadfeatures1);
	drawfeatures(dstyemp, quadfeatures2);
	imshow("source", srcyemp);
	imshow("target", dstyemp);
	waitKey(20);*/


	if ((quadfeatures1.size()<4) || (quadfeatures2.size()<4))
	{
		return false;
	}
	if (quadfeatures1.size() != quadfeatures2.size())
	{
		return false;
	}
	features_img1.resize(0);
	features_img2.resize(0);

	vector<uchar> mask;

	findHomography(cv::Mat(quadfeatures1), cv::Mat(quadfeatures2), mask, CV_RANSAC, threshold);
	for (int k = 0; k < mask.size(); k++){
		if (mask[k] == 1){
			features_img1.push_back(quadfeatures1[k]);
			features_img2.push_back(quadfeatures2[k]);
		}
	}
	//cout << features_img1.size() << endl;
	//cout << features_img2.size() << endl;

	cv::Mat srcyemp1 = src.clone();
	cv::Mat dstyemp1 = dst.clone();
	drawfeatures(srcyemp1, features_img1);
	drawfeatures(dstyemp1, features_img2);
	imshow("source1", srcyemp1);
	imshow("target1", dstyemp1);
	waitKey(20);

	return true;

}


void GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2){

	if (features_img1.size() > 10){
		vector<cv::Point2f> temp1, temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for (int i = 0; i < features_img1.size(); i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		vector<uchar> mask;
		cv::findHomography(cv::Mat(temp1), cv::Mat(temp2), mask, CV_RANSAC, 20);

		for (int k = 0; k < mask.size(); k++){
			if (mask[k] == 1){
				features_img1.push_back(temp1[k]);
				features_img2.push_back(temp2[k]);
			}
		}
	}
	else{
		return;
	}
}