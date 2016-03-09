#include "User_System.h"
#include "CommonOperate.h"
#include "MyDebug.h"
#include "save.h"
#include "MyImageStitch.h"

using namespace std;
using namespace cv;

int main()
{
	if (freopen("output.txt", "w", stdout) == NULL)
		fprintf(stderr, "error redirecting stdout\n");

	vector<cv::Mat>images,mask,mask_graph;
	cv::Mat im0 = imread("image_warpedAftercompensator_0.png");
	cv::Mat im1 = imread("image_warpedAftercompensator_1.png");
	cv::Mat mask0 = imread("mask_compensator_0.png",0);
	cv::Mat mask1 = imread("mask_compensator_1.png", 0);
	cv::Mat mask_graph_0 = imread("mask_graphcut_0.png", 0);
	cv::Mat mask_graph_1 = imread("mask_graphcut_1.png", 0);
	printf("Hello world\n");
	images.push_back(im0);
	images.push_back(im1);
	mask.push_back(mask0);
	mask.push_back(mask1);
	mask_graph.push_back(mask_graph_0);
	mask_graph.push_back(mask_graph_1);



	vector<cv::Point>corners(2);
	cv::Point pt0(0, 0);
	cv::Point pt1(212, 12);
	corners[0] = pt0;
	corners[1] = pt1;



	cv::Mat result = ImageStitch(images,mask,mask_graph,corners);

	imwrite("result.png", result);
	waitKey();


	
}
