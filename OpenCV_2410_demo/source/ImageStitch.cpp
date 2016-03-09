#include "ImageStitch.h"
#include "LaplacianBlending.h"
#include "mywarp.h"
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
using namespace detail;

int flagsave = 0;

void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result, vector<cv::Mat> imagemask)
{
	Point pt(0.0, 0.0);
	vector<Point>corners(2);
	corners[0] = pt;
	corners[1] = pt;

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	compensator->feed(corners, images, imagemask);


	vector<Size> sizes(2);
	sizes[0] = images[0].size();
	sizes[1] = images[1].size();

	int blend_type = Blender::MULTI_BAND;
	Ptr<Blender> blender;
	float blend_strength = 5;

	for (int img_idx = 0; img_idx < 2; ++img_idx){

		Mat img_warped = images[img_idx].clone();
		compensator->apply(img_idx, corners[img_idx], img_warped, imagemask[img_idx]);
		img_warped.convertTo(img_warped, CV_16S);


		if (blender.empty())
		{
			blender = Blender::createDefault(blend_type, false);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, false);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
				fb->setSharpness(1.f / blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			blender->prepare(corners, sizes);
		}
	
		// Blend the current image
		blender->feed(img_warped, masks[img_idx], corners[img_idx]);
	}

	Mat result_mask;
	blender->blend(result, result_mask);


	
}


void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result)
{
	Point pt(0.0 - images[0].cols, 0.0 - images[0].rows);
	vector<Point>corners(2);
	corners[0] = pt;
	corners[1] = pt;
	vector<Size> sizes(2);
	sizes[0] = images[0].size();
	sizes[1] = images[1].size();
	images[0].convertTo(images[0], CV_16S);
	images[1].convertTo(images[1], CV_16S);
	int blend_type = Blender::MULTI_BAND;
	Ptr<Blender> blender;
	float blend_strength = 5;
	if (blender.empty())
	{
		blender = Blender::createDefault(blend_type, false);
		Size dst_sz = resultRoi(corners, sizes).size();
		float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
		if (blend_width < 1.f)
			blender = Blender::createDefault(Blender::NO, false);
		else if (blend_type == Blender::MULTI_BAND)
		{
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			LOGLN("Multi-band blender, number of bands: " << mb->numBands());
		}
		else if (blend_type == Blender::FEATHER)
		{
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
			fb->setSharpness(1.f / blend_width);
			LOGLN("Feather blender, sharpness: " << fb->sharpness());
		}
		blender->prepare(corners, sizes);
	}
	for (int img_idx = 0; img_idx < 2; ++img_idx){
		//compensator->apply(img_idx, corners[img_idx], images[img_idx], masks[img_idx]);
		// Blend the current image
		blender->feed(images[img_idx], masks[img_idx], corners[img_idx]);
	}

	Mat result_mask;
	blender->blend(result, result_mask);



}


void MyImageBlend(vector<cv::Mat>masks, vector<cv::Mat>images, cv::Mat &result,vector<cv::Point>corners)
{
	vector<Size> sizes(2);
	sizes[0] = images[0].size();
	sizes[1] = images[1].size();
	images[0].convertTo(images[0], CV_16S);
	images[1].convertTo(images[1], CV_16S);
	int blend_type = Blender::MULTI_BAND;
	Ptr<Blender> blender;
	float blend_strength = 5;
	if (blender.empty())
	{
		blender = Blender::createDefault(blend_type, false);
		Size dst_sz = resultRoi(corners, sizes).size();
		float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
		if (blend_width < 1.f)
			blender = Blender::createDefault(Blender::NO, false);
		else if (blend_type == Blender::MULTI_BAND)
		{
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			LOGLN("Multi-band blender, number of bands: " << mb->numBands());
		}
		else if (blend_type == Blender::FEATHER)
		{
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
			fb->setSharpness(1.f / blend_width);
			LOGLN("Feather blender, sharpness: " << fb->sharpness());
		}
		blender->prepare(corners, sizes);
	}
	for (int img_idx = 0; img_idx < 2; ++img_idx){
		//compensator->apply(img_idx, corners[img_idx], images[img_idx], masks[img_idx]);
		// Blend the current image
		blender->feed(images[img_idx], masks[img_idx], corners[img_idx]);
	}

	Mat result_mask;
	blender->blend(result, result_mask);



}

void findcorner(cv::Mat image, vector<cv::Point2f>&corners,int num)
{
	image.convertTo(image, CV_8UC1);
	goodFeaturesToTrack(image, corners, num, 0.05, 20);
}


cv::Point2f findMiddlePoint(cv::Mat image)
{
	vector<cv::Point2f>corners;
	image.convertTo(image, CV_8UC1);
	goodFeaturesToTrack(image, corners, 4, 0.05, 20);
	Point2f middlePoint((corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4, (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4);
	return middlePoint;
}

void getTopLeftcorner(vector<cv::Point2f>corner, vector<cv::Point2f>&cornersrank)
{
	cornersrank.resize(corner.size());
	vector<double>Xset(4);
	vector<double>Yset(4);

	Point2f tl, tr, bl, br;
	int tlindex, trindex, blindex, brindex;
	int numindex = 3 + 2 + 1 + 0;

	for (int i = 0; i < corner.size(); i++)
	{
		Xset[i] = corner[i].x;
		Yset[i] = corner[i].y;
	}
	vector<int>XsetIndex;
	vector<int>YsetIndex;
	SelectionSort(Xset, XsetIndex, 1);
	SelectionSort(Yset, YsetIndex, 1);
	if (corner[XsetIndex[0]].y < corner[XsetIndex[1]].y)
	{
		tl = corner[XsetIndex[0]];
		bl = corner[XsetIndex[1]];
		tlindex = XsetIndex[0];
		blindex = XsetIndex[1];
	}
	else
	{
		bl = corner[XsetIndex[0]];
		tl = corner[XsetIndex[1]];
		tlindex = XsetIndex[1];
		blindex = XsetIndex[0];
	}

	if (corner[YsetIndex[0]].x < corner[YsetIndex[1]].x)
	{
		tl = corner[YsetIndex[0]];
		tr = corner[YsetIndex[1]];
		tlindex = YsetIndex[0];
		trindex = YsetIndex[1];
	}
	else
	{
		tl = corner[YsetIndex[1]];
		tr = corner[YsetIndex[0]];
		tlindex = YsetIndex[1];
		trindex = YsetIndex[0];
	}


	br = corner[numindex - tlindex - trindex - blindex];

	cornersrank[0] = tl;
	cornersrank[1] = tr;
	cornersrank[2] = bl;
	cornersrank[3] = br;

}



//void ImageStitch(vector<cv::Point> corners, vector<cv::Mat> images_warpngaps, vector<cv::Mat> mask_warpngaps,
//	vector<cv::Point2f> maskcorner0, vector<cv::Point2f> maskcorner1,cv::Mat& result)
//{
//	cv::Mat texture(images_warpngaps[0].size(), CV_8UC3, Scalar(0, 0, 0));
//	cv::Mat coarseresult, overlapmask;
//	vector<cv::Mat> overlap(2);
//	vector<cv::Mat> overlap_save(2);
//	myimagefusion(images_warpngaps[0], images_warpngaps[1], coarseresult, overlap[0], overlap[1], 0, texture);
//	overlap_save[0] = overlap[0].clone();
//	overlap_save[1] = overlap[1].clone();
//	imwrite("overlap0.jpg", overlap[0]);
//	imwrite("overlap1.jpg", overlap[1]);
//	getoverlapmask(images_warpngaps[0], images_warpngaps[1], overlapmask);
//	vector<cv::Point2f>vertex, sortvertex;
//	imwrite("overlapmask.jpg", overlapmask);
//	findcorner(overlapmask, vertex);
//	getTopLeftcorner(vertex, sortvertex);//得到按照tl,tr,bl,br排序的顶点序列sortvertex
//	int width = 10;
//	vector<cv::Point2f> sortvertixnew(4);
//	vector<cv::Point2f> sortvertixnew_normalrank(4);
//	sortvertixnew[0] = Point2f(sortvertex[0].x + width, sortvertex[0].y + width);
//	sortvertixnew[1] = Point2f(sortvertex[1].x - width, sortvertex[1].y + width);
//	sortvertixnew[2] = Point2f(sortvertex[3].x - width, sortvertex[3].y - width);
//	sortvertixnew[3] = Point2f(sortvertex[2].x + width, sortvertex[2].y - width);//得到缩进后的新的mask的顶点坐标，按照tl,tr,br,bl排列
//	sortvertixnew_normalrank[0] = sortvertixnew[0];
//	sortvertixnew_normalrank[1] = sortvertixnew[1];
//	sortvertixnew_normalrank[2] = sortvertixnew[3];
//	sortvertixnew_normalrank[3] = sortvertixnew[2];
//
//
//	cv::Mat smallOverlapMask = cv::Mat::zeros(overlapmask.size(), CV_8UC1);
//	MyDrawPolygon(smallOverlapMask, sortvertixnew);                           //获取缩进后的mask
//
//	vector<cv::Mat>appendixmask(4);
//	getAppendixMask(overlapmask, smallOverlapMask, appendixmask, sortvertex);
//
//	vector<cv::Mat>masktofindseam(2);
//	masktofindseam[0] = smallOverlapMask.clone();
//	masktofindseam[1] = smallOverlapMask.clone();
//	overlap[0].convertTo(overlap[0], CV_32FC3);
//	overlap[1].convertTo(overlap[1], CV_32FC3);
//	Ptr<SeamFinder> seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
//	if (seam_finder.empty())
//	{
//		cout << "Can't create the following seam finder" << "'\n";
//		exit(0);
//	}
//	seam_finder->find(overlap, corners, masktofindseam);
//	//imwrite("masktofindseam0.jpg", masktofindseam[0]);
//	//imwrite("masktofindseam1.jpg", masktofindseam[1]);
//	vector<cv::Point2f>middlepoints(2);
//	cv::Point2f mp0((maskcorner0[0].x + maskcorner0[1].x + maskcorner0[2].x + maskcorner0[3].x) / 4, (maskcorner0[0].y + maskcorner0[1].y + maskcorner0[2].y + maskcorner0[3].y) / 4);
//	cv::Point2f mp1((maskcorner1[0].x + maskcorner1[1].x + maskcorner1[2].x + maskcorner1[3].x) / 4, (maskcorner1[0].y + maskcorner1[1].y + maskcorner1[2].y + maskcorner1[3].y) / 4);
//	middlepoints[0] = mp0;
//	middlepoints[1] = mp1;
//	//imwrite("overlapmask.jpg", overlapmask);
//	vector<cv::Mat>resultmask(2);
//	mask_warpngaps[0].convertTo(mask_warpngaps[0], CV_8UC1);
//	mask_warpngaps[1].convertTo(mask_warpngaps[1], CV_8UC1);
//	rebuidmask(masktofindseam, appendixmask, sortvertixnew_normalrank, middlepoints, resultmask, overlapmask,mask_warpngaps);
//	/*imwrite("overlapmask.jpg", overlapmask);
//	imwrite("resulymask0.jpg",resultmask[0]);
//	imwrite("resulymask1.jpg", resultmask[1]);*/
//	
//
//
//	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
//	compensator->feed(corners, images_warpngaps, resultmask);
//
//	for (int img_idx = 0; img_idx < images_warpngaps.size(); img_idx++)
//	{
//		compensator->apply(img_idx, corners[img_idx], images_warpngaps[img_idx], resultmask[img_idx]);
//	}
//
//
//	cv::Mat result_temp;
//	MyImageBlend(resultmask, images_warpngaps, result_temp);
//	result_temp.convertTo(result_temp, CV_8UC3);
//	result_temp.copyTo(result);
//	
//}

void ImageStitch(vector<cv::Point> corners, vector<cv::Mat> images_warpngaps, vector<cv::Mat> mask_warpngaps,
	vector<cv::Point2f> maskcorner0, vector<cv::Point2f> maskcorner1, cv::Mat& result)
{
	cv::Mat texture(images_warpngaps[0].size(), CV_8UC3, Scalar(0, 0, 0));
	cv::Mat coarseresult, overlapmask;
	vector<cv::Mat> overlap(2);
	myimagefusion(images_warpngaps[0], images_warpngaps[1], coarseresult, overlap[0], overlap[1], 0, texture);
	getoverlapmask(images_warpngaps[0], images_warpngaps[1], overlapmask);
	vector<Point2f> overlapmaskcorners;
	findcorner(overlapmask, overlapmaskcorners, 4);
	
	vector<Point2f> mask0vertex(4);
	vector<Point2f> mask1vertex(4);



	int width = 25;



	mask0vertex[0] = Point2f(maskcorner0[0].x + width, maskcorner0[0].y + width);
	mask0vertex[1] = Point2f(maskcorner0[1].x - width, maskcorner0[1].y + width);
	mask0vertex[2] = Point2f(maskcorner0[3].x - width, maskcorner0[3].y - width);
	mask0vertex[3] = Point2f(maskcorner0[2].x + width, maskcorner0[2].y - width);//得到缩进后的新的mask的顶点坐标，按照tl,tr,br,bl排列
	mask1vertex[0] = Point2f(maskcorner1[0].x + width, maskcorner1[0].y + width);
	mask1vertex[1] = Point2f(maskcorner1[1].x - width, maskcorner1[1].y + width);
	mask1vertex[2] = Point2f(maskcorner1[3].x - width, maskcorner1[3].y - width);
	mask1vertex[3] = Point2f(maskcorner1[2].x + width, maskcorner1[2].y - width);//得到缩进后的新的mask的顶点坐标，按照tl,tr,br,bl排列
	cv::Mat Mask0 = cv::Mat::zeros(mask_warpngaps[0].size(), CV_8UC1);
	MyDrawPolygon(Mask0, mask0vertex);                           //获取缩进后的mask
	cv::Mat Mask1 = cv::Mat::zeros(mask_warpngaps[1].size(), CV_8UC1);
	MyDrawPolygon(Mask1, mask1vertex);                           //获取缩进后的mask
	cv::Mat standmask(Mask0.size(), CV_8UC1);
	//int isMask0use=determineWhichMaskToUse(Mask0, Mask1, overlapmaskcorners);
	int isMask0use = 0;
	if (isMask0use==1)  //use mask0
	{
		
		mask_warpngaps[0] = Mask0;
		standmask.setTo(Scalar::all(0));
		standmask = mask_warpngaps[0] | mask_warpngaps[1];
		mask_warpngaps[1] = standmask - mask_warpngaps[0];
		
	}
	else if (isMask0use == 0)
	{
		mask_warpngaps[1] = Mask1;
		standmask.setTo(Scalar::all(0));
		standmask = mask_warpngaps[0] | mask_warpngaps[1];
		mask_warpngaps[0] = standmask - mask_warpngaps[1];
	}

	cv::Mat result_temp;
	MyImageBlend(mask_warpngaps, images_warpngaps, result_temp);
	result_temp.convertTo(result_temp, CV_8UC3);
	
	
	result_temp.copyTo(result);

}


void ImageStitch(vector<cv::Mat> images_warpngaps, vector<cv::Mat> mask_warpngaps,cv::Mat& result)
{
	cv::Mat standmask(mask_warpngaps[1].size(), CV_8UC1);
	standmask.setTo(Scalar::all(0));
	standmask = mask_warpngaps[0] | mask_warpngaps[1];
	mask_warpngaps[0] = standmask - mask_warpngaps[1];
	cv::Mat result_temp;
	MyImageBlend(mask_warpngaps, images_warpngaps, result_temp);
	result_temp.convertTo(result_temp, CV_8UC3);
	result_temp.copyTo(result);

}


void getAppendixMask(cv::Mat mask, cv::Mat smallmask, vector<cv::Mat>&appendixmask, vector<cv::Point2f> maskvertex)
{
	for (int i = 0; i < appendixmask.size(); ++i)
	{
		appendixmask[i].create(mask.size(), CV_8U);
		appendixmask[i].setTo(Scalar::all(0));
	}
	cv::Mat mask_tl, mask_tr, mask_bl, mask_br;
	cv::Mat submask = mask - smallmask;
	cv::Mat trianglemask_tl, trianglemask_tr, trianglemask_bl, trianglemask_br;
	vector<cv::Point2f>triangleVertex(5);
	triangleVertex[0] = cv::Point2f(mask.cols - 1, 0.0);
	triangleVertex[1] = maskvertex[1];
	triangleVertex[2] = maskvertex[2];
	triangleVertex[3] = cv::Point2f(0.0, mask.rows - 1);
	triangleVertex[4] = cv::Point2f(0, 0);

	trianglemask_tl.create(mask.size(), CV_8U);
	trianglemask_tl.setTo(Scalar::all(0));
	MyDrawPolygon(trianglemask_tl, triangleVertex);
	mask_tl = submask&trianglemask_tl;

	triangleVertex[0] = cv::Point2f(0.0, 0.0);
	triangleVertex[1] = maskvertex[0];
	triangleVertex[2] = maskvertex[3];
	triangleVertex[3] = cv::Point2f(mask.cols - 1, mask.rows - 1);
	triangleVertex[4] = cv::Point2f(mask.cols - 1,0);
	trianglemask_tr.create(mask.size(), CV_8U);
	trianglemask_tr.setTo(Scalar::all(0));
	MyDrawPolygon(trianglemask_tr, triangleVertex);
	mask_tr = submask&trianglemask_tr;


	mask_bl = submask - mask_tr;
	mask_br = submask - mask_tl;

	

	appendixmask[0] = mask_tl;
	appendixmask[1] = mask_tr;
	appendixmask[2] = mask_bl;
	appendixmask[3] = mask_br;

	
}



void rebuidmask(vector<cv::Point2f>overlapvertex,vector<cv::Point2f>corners, vector<cv::Mat>&resultmask, cv::Mat overlapmask, vector<cv::Mat>mask_warpngaps)
{
	mask_warpngaps[0] = mask_warpngaps[0] - overlapmask;
	mask_warpngaps[1] = mask_warpngaps[1] - overlapmask;

}



int determineWhichMaskToUse(cv::Mat mask0, cv::Mat mask1, vector<cv::Point2f>overlapcorners)
{
	int num1 = 0;
	int num2 = 0;
	int flag = 0;
	for (int i = 0; i < overlapcorners.size();i++)
	{
		cout << overlapcorners[i] << endl;
		if (mask0.at<uchar>(overlapcorners[i].y, overlapcorners[i].x)!=0)
		{
			num1++;
		}
		if (mask1.at<uchar>(overlapcorners[i].y, overlapcorners[i].x) != 0)
		{
			num2++;
		}
	}
	cout << num1 << endl;
	cout << num2<<endl;
	if (num1 > num2)
	{
		flag=0;  //取mask1
	}
	else if (num1<num2)
		flag = 1;//取mask0
	else if (num1 == num2)
		flag = flagsave;   //和上一次保持一致

	flagsave = flag;
	return flag;
}


//void rebuidmask(vector<cv::Mat>masktofindseam,vector<cv::Mat>appendixmask,vector<cv::Point2f>overlapvertex,
//	vector<cv::Point2f>corners, vector<cv::Mat>&resultmask, cv::Mat overlapmask, vector<cv::Mat>mask_warpngaps)
//{
//	mask_warpngaps[0] = mask_warpngaps[0] - overlapmask;
//	mask_warpngaps[1] = mask_warpngaps[1] - overlapmask;
//
//	resultmask.resize(2);
//	cv::Mat mask_seam = masktofindseam[1];////////////////////////这里是因为masktofindseam[1]用的是单位阵，其他的可能还是要按照变换后原点的位置返回去
//	imwrite("mask_seam.jpg", mask_seam);
//	if ((corners[0].x>corners[1].x) && (corners[0].y<corners[1].y))    //warp top right
//	{
//		cv::Point2f judgepoint(overlapvertex[2].x + 10, overlapvertex[2].y - 10);
//		int judgevalue = mask_seam.at<uchar>(judgepoint.y, judgepoint.x);
//		if (judgevalue == 255)
//		{
//			resultmask[1] = mask_seam + appendixmask[2] + mask_warpngaps[1];
//			resultmask[0] = overlapmask - resultmask[1] + mask_warpngaps[0];
//		}
//		else if (judgevalue == 0)
//		{
//			resultmask[0] = mask_seam + appendixmask[1] + mask_warpngaps[0];
//			resultmask[1] = overlapmask - resultmask[0] + mask_warpngaps[1];
//		}
//	}
//	if ((corners[0].x > corners[1].x) && (corners[0].y > corners[1].y))   //warp bottom right
//	{
//		cv::Point2f judgepoint(overlapvertex[3].x - 10, overlapvertex[3].y - 10);
//		int judgevalue = mask_seam.at<uchar>(judgepoint.y, judgepoint.x);
//		if (judgevalue == 255)
//		{
//			resultmask[0] = mask_seam + appendixmask[3] + mask_warpngaps[0];
//			resultmask[1] = overlapmask - resultmask[0] + mask_warpngaps[1];
//		}
//		else if (judgevalue == 0)
//		{
//			resultmask[1] = mask_seam + appendixmask[0] + mask_warpngaps[1];
//			resultmask[0] = overlapmask - resultmask[1] + mask_warpngaps[0];
//		}
//	}
//	if ((corners[0].x < corners[1].x) && (corners[0].y < corners[1].y))    //warp top left
//	{
//		cv::Point2f judgepoint(overlapvertex[3].x - 10, overlapvertex[3].y - 10);
//		int judgevalue = mask_seam.at<uchar>(judgepoint.y, judgepoint.x);
//		if (judgevalue == 255)
//		{
//			resultmask[1] = mask_seam + appendixmask[3] + mask_warpngaps[1];
//			resultmask[0] = overlapmask - resultmask[1] + mask_warpngaps[0];
//		}
//		else if (judgevalue == 0)
//		{
//			resultmask[0] = mask_seam + appendixmask[0] + mask_warpngaps[0];
//			resultmask[1] = overlapmask - resultmask[0] + mask_warpngaps[1];
//		}
//	}
//	if ((corners[0].x < corners[1].x) && (corners[0].y > corners[1].y))   //warp bottom left
//	{
//		cv::Point2f judgepoint(overlapvertex[2].x + 10, overlapvertex[2].y - 10);
//		int judgevalue = mask_seam.at<uchar>(judgepoint.y, judgepoint.x);
//		if (judgevalue == 255)
//		{
//			resultmask[0] = mask_seam + appendixmask[2] + mask_warpngaps[0];
//			resultmask[1] = overlapmask - resultmask[0] + mask_warpngaps[1];
//		}
//		else if (judgevalue == 0)
//		{
//			resultmask[1] = mask_seam + appendixmask[1] + mask_warpngaps[1];
//			resultmask[0] = overlapmask - resultmask[1] + mask_warpngaps[0];
//		}
//	}
//}