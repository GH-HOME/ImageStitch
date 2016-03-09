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

	Mat_< float> blendMask;



	vector<Mat_<Vec3f> > topLapPyr, downLapPyr, resultLapPyr; //Laplacian Pyramids  

	Mat topHighestLevel, downHighestLevel, resultHighestLevel;

	vector<Mat_<Vec3f> > maskGaussianPyramid; //masks are 3-channels for easier multiplication with RGB  



	int levels;



	//������e��?��t

	void buildPyramids() {

		//��?��y��?��a�� top����?top ,topLapPyr����?top��?laplacian��?pyr,��?topHighestLevel����?��?��?��?��?��?��?��?˹1��e��?��t

		buildLaplacianPyramid(top, topLapPyr, topHighestLevel);

		buildLaplacianPyramid(down, downLapPyr, downHighestLevel);

		buildGaussianPyramid();

	}



	//����gauss��e��?��t

	void buildGaussianPyramid() {//��e��?��t����YΪaÿ?һ?��?��?��ģ  

		assert(topLapPyr.size() > 0);



		maskGaussianPyramid.clear();

		Mat currentImg;

		//blendMask����?����?

		cvtColor(blendMask, currentImg, CV_GRAY2BGR); //store color img of blend mask into maskGaussianPyramid  

		maskGaussianPyramid.push_back(currentImg); //0-level  



		currentImg = blendMask;

		for (int l = 1; l<levels + 1; l++) {

			Mat _down;

			if (topLapPyr.size() > l)

				pyrDown(currentImg, _down, topLapPyr[l].size());

			else

				pyrDown(currentImg, _down, topHighestLevel.size()); //lowest level  



			Mat down;

			cvtColor(_down, down, CV_GRAY2BGR);

			maskGaussianPyramid.push_back(down); //add color blend mask into mask Pyramid  

			currentImg = _down;

		}

	}



	//����laplacian��e��?��t

	void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& HighestLevel) {

		lapPyr.clear();

		Mat currentImg = img;

		for (int l = 0; l < levels; l++) {

			Mat down, up;

			pyrDown(currentImg, down);

			pyrUp(down, up, currentImg.size());

			Mat lap = currentImg - up;  //��?����?����?��D��?

			lapPyr.push_back(lap);

			currentImg = down;

		}

		currentImg.copyTo(HighestLevel);

	}



	Mat reconstructImgFromLapPyramid() {

		//��?����laplacianͼ?��?ƴ�ɵ�?resultLapPyr��e��?��t��Dÿ?һ?��?  

		//����?��?��?��?ֵ��?����ӣ�?����?blendͼ?��?���?  

		Mat currentImg = resultHighestLevel;

		for (int l = levels - 1; l >= 0; l--) {

			Mat up;

			pyrUp(currentImg, up, resultLapPyr[l].size());

			currentImg = up + resultLapPyr[l];

		}

		return currentImg;

	}



	void blendLapPyrs() {

		//��?��?ÿ?��?��e��?��t��Dֱ����?������?ͼ?Laplacian��?��?ƴ�ɵ�?ͼ?��?resultLapPyr  

		//һ?��?��?һ?��?����?����a��?��?��?��?��?��?�� ��?������ģ��?��?ʽ?��?��D��?.

		resultHighestLevel = topHighestLevel.mul(maskGaussianPyramid.back()) +

			downHighestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());

		for (int l = 0; l < levels; l++) {

			Mat A = topLapPyr[l].mul(maskGaussianPyramid[l]);

			Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];

			Mat B = downLapPyr[l].mul(antiMask);

			Mat_<Vec3f> blendedLevel = A + B;

			resultLapPyr.push_back(blendedLevel);

		}

	}



public:

	LaplacianBlending(const Mat_<Vec3f>& _top, const Mat_<Vec3f>& _down, const Mat_< float>& _blendMask, int _levels) ://ȱʡ?��y��Y��?ʹ1��? LaplacianBlending lb(l,r,m,4);  

		top(_top), down(_down), blendMask(_blendMask), levels(_levels)

	{

		assert(_top.size() == _down.size());

		assert(_top.size() == _blendMask.size());

		buildPyramids();  //����laplacian��e��?��t��gauss��e��?��t

		blendLapPyrs();   //��?���ҽ�e��?��t�ں�?��Ϊaһ?��?ͼ?Ƭ?  

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

Mat LaplacianBlend(const Mat_<Vec3f>& t, const Mat_<Vec3f>& d, const Mat_< float>& m) {

	LaplacianBlending lb(t, d, m, 4);

	return lb.blend();

}


Mat LaplacianBlend(const Mat source, const Mat target, Mat mask) {

	Mat_<Vec3f> t; source.convertTo(t, CV_32F, 1.0 / 255.0); //Vec3f��ʾ?��D��y��?ͨ����?�� l[row][column][depth]  

	Mat_<Vec3f> d; target.convertTo(d, CV_32F, 1.0 / 255.0);
	Mat_< float> m(mask.size(), 0.0);
	setMask(mask, m);
	LaplacianBlending lb(t, d, m, 4);
	cv::Mat blend = lb.blend();
	cv::Mat result_Mat(blend.size(), CV_8UC3);
	blend.convertTo(result_Mat, CV_8UC3, 255);
	return result_Mat;


}





#endif