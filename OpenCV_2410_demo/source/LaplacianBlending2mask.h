#include "User_System.h"
#include "CommonOperate.h"

using namespace std;
using namespace cv;

#ifndef __LAPLACE_BLEND_H
#define __LAPLACE_BLEND_H


class LaplacianBlending {

private:

	Mat_<Vec3f> top;

	Mat_<Vec3f> down;

	Mat_< float> blendMask1;
	Mat_< float> blendMask2;




	vector<Mat_<Vec3f> > topLapPyr, downLapPyr, resultLapPyr; //Laplacian Pyramids  

	Mat topHighestLevel, downHighestLevel, resultHighestLevel;

	vector<Mat_<Vec3f> > maskGaussianPyramid1; //masks are 3-channels for easier multiplication with RGB  
	vector<Mat_<Vec3f> > maskGaussianPyramid2; //masks are 3-channels for easier multiplication with RGB  



	int levels;



	//创建金e字?塔t

	void buildPyramids() {

		//参?数y的?解a释 top就是?top ,topLapPyr就是?top的?laplacian的?pyr,而?topHighestLevel保存?的?是?最?高?端?的?高?斯1金e字?塔t

		buildLaplacianPyramid(top, topLapPyr, topHighestLevel);

		buildLaplacianPyramid(down, downLapPyr, downHighestLevel);

		buildGaussianPyramidTop();

		buildGaussianPyramidDown();

	}



	//创建gauss金e字?塔t

	void buildGaussianPyramidTop() {//金e字?塔t内容Y为a每?一?层?的?掩模  

		assert(topLapPyr.size() > 0);



		maskGaussianPyramid1.clear();

		Mat currentImg;

		//blendMask就是?掩码?

		cvtColor(blendMask1, currentImg, CV_GRAY2BGR); //store color img of blend mask into maskGaussianPyramid  

		maskGaussianPyramid1.push_back(currentImg); //0-level  



		currentImg = blendMask1;

		for (int l = 1; l<levels + 1; l++) {

			Mat _down;

			if (topLapPyr.size() > l)

				pyrDown(currentImg, _down, topLapPyr[l].size());

			else

				pyrDown(currentImg, _down, topHighestLevel.size()); //lowest level  



			Mat down;

			cvtColor(_down, down, CV_GRAY2BGR);

			maskGaussianPyramid1.push_back(down); //add color blend mask into mask Pyramid  

			currentImg = _down;

		}

	}


	void buildGaussianPyramidDown() {//金e字?塔t内容Y为a每?一?层?的?掩模  

		assert(downLapPyr.size() > 0);



		maskGaussianPyramid2.clear();

		Mat currentImg;

		//blendMask就是?掩码?

		cvtColor(blendMask2, currentImg, CV_GRAY2BGR); //store color img of blend mask into maskGaussianPyramid  

		maskGaussianPyramid2.push_back(currentImg); //0-level  



		currentImg = blendMask2;

		for (int l = 1; l<levels + 1; l++) {

			Mat _down;

			if (downLapPyr.size() > l)

				pyrDown(currentImg, _down, downLapPyr[l].size());

			else

				pyrDown(currentImg, _down, downHighestLevel.size()); //lowest level  



			Mat down;

			cvtColor(_down, down, CV_GRAY2BGR);

			maskGaussianPyramid2.push_back(down); //add color blend mask into mask Pyramid  

			currentImg = _down;

		}

	}


	//创建laplacian金e字?塔t

	void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& HighestLevel) {

		lapPyr.clear();

		Mat currentImg = img;

		for (int l = 0; l < levels; l++) {

			Mat down, up;

			pyrDown(currentImg, down);

			pyrUp(down, up, currentImg.size());

			Mat lap = currentImg - up;  //存?储的?就是?残D差?

			lapPyr.push_back(lap);

			currentImg = down;

		}

		currentImg.copyTo(HighestLevel);

	}



	Mat reconstructImgFromLapPyramid() {

		//将?左右laplacian图?像?拼成的?resultLapPyr金e字?塔t中D每?一?层?  

		//从上?到?下?插?值放?大并相加，?即得?blend图?像?结果?  

		Mat currentImg = resultHighestLevel;

		for (int l = levels - 1; l >= 0; l--) {

			Mat up;

			pyrUp(currentImg, up, resultLapPyr[l].size());

			currentImg = up + resultLapPyr[l];

		}

		return currentImg;

	}



	void blendLapPyrs() {

		//获?得?每?层?金e字?塔t中D直接用?左右两?图?Laplacian变?换?拼成的?图?像?resultLapPyr  

		//一?半?的?一?半?就是?在这a个?地?方?计?算?的?。 是?基于掩模的?方?式?进?行D的?.

		resultHighestLevel = topHighestLevel.mul(maskGaussianPyramid1.back()) +

			downHighestLevel.mul(maskGaussianPyramid2.back());

		for (int l = 0; l < levels; l++) {

			Mat A = topLapPyr[l].mul(maskGaussianPyramid1[l]);



			Mat B = downLapPyr[l].mul(maskGaussianPyramid2[l]);

			Mat_<Vec3f> blendedLevel = A + B;

			resultLapPyr.push_back(blendedLevel);

		}

	}



public:

	LaplacianBlending(const Mat_<Vec3f>& _top, const Mat_<Vec3f>& _down, const Mat_< float>& _blendMask1, const Mat_< float>& _blendMask2, int _levels) ://缺省?数y据Y，?使1用? LaplacianBlending lb(l,r,m,4);  

		top(_top), down(_down), blendMask1(_blendMask1), blendMask2(_blendMask2), levels(_levels)

	{

		assert(_top.size() == _down.size());

		assert(_top.size() == _blendMask1.size());

		assert(_down.size() == _blendMask2.size());

		buildPyramids();  //创建laplacian金e字?塔t和gauss金e字?塔t

		blendLapPyrs();   //将?左右金e字?塔t融合?成为a一?个?图?片?  

	};



	Mat blend() {

		return reconstructImgFromLapPyramid();//reconstruct Image from Laplacian Pyramid  

	}

};

void setMask(cv::Mat Mask, Mat_<float> &resultMask)
{
	for (int i = 0; i < Mask.rows; i++)
	{
		for (int j = 0; j < Mask.cols; j++)
		{
			if (Mask.at<uchar>(i, j) != 0)
			{
				resultMask(i, j) = 1.0;
			}
			else
			{
				resultMask(i, j) = 0.0;
			}
		}
	}
}

Mat LaplacianBlend(const Mat_<Vec3f>& t, const Mat_<Vec3f>& d, const Mat_< float>& m1, const Mat_< float>& m2) {

	LaplacianBlending lb(t, d, m1, m2, 4);

	return lb.blend();

}


Mat LaplacianBlend(const Mat source, const Mat target, Mat mask1, Mat mask2) {

	Mat_<Vec3f> t; source.convertTo(t, CV_32F, 1.0 / 255.0); //Vec3f表示?有D三y个?通道，?即 l[row][column][depth]  

	Mat_<Vec3f> d; target.convertTo(d, CV_32F, 1.0 / 255.0);
	Mat_< float> m1(mask1.size(), 0.0);
	Mat_< float> m2(mask2.size(), 0.0);
	setMask(mask1, m1);
	setMask(mask2, m2);
	LaplacianBlending lb(t, d, m1, m2, 4);
	cv::Mat blend = lb.blend();
	cv::Mat result_Mat(blend.size(), CV_8UC3);
	blend.convertTo(result_Mat, CV_8UC3, 255);
	return result_Mat;


}





#endif