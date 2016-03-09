#pragma once
#include "User_System.h"
#include "Mesh.h"

void mySURF(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2);
void GlobalOutLinerRejectorOneIteration_SURF(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold);
bool QuadRansac(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, Quad Imagequad, double threshold);
void ImageSizeRejectorOneIteration_SURF(int img_Height, int img_Width, vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2);
void mySift(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2);
void MyRANSAC(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold, Size imageSize);
bool QuadRansac(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, Quad Imagequad, double threshold, cv::Mat src, cv::Mat dst);
void MyRANSAC(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2, double threshold, Size imageSize, cv::Mat src, cv::Mat dst);
void GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2);
