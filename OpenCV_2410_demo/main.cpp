#include "User_System.h"
#include "CommonOperate.h"
#include "MyDebug.h"
#include "save.h"
#include "MyImageStitch.h"

using namespace std;
using namespace cv;

cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1, int index);

cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1);

int readvideo(char*videoname,vector<cv::Mat>&frames)
{
	printf("Read Video\n");
	cout << "now read video" << videoname << endl;
	cv::VideoCapture capture(videoname);
	// check if video successfully opened
	if (!capture.isOpened())
	{
		cerr << "The video can not open!";
		return -1;
		exit(0);
	}
	
	cv::Mat frame; // current video frame
	printf("extract frames...");
	int frame_count = 0;
	while (capture.read(frame))
	{
		cv::Mat frame_copy = frame.clone();
		frames.push_back(frame_copy);
		printf("%04d\b\b\b\b", frame_count);
		frame_count++;
	}
	capture.release();
	printf("Video Capture Done!\n");
	return 0;

}

void writevideo(char*videoname, vector<cv::Mat>frames)
{
	printf("Rendering...");
	cv::VideoWriter outVideoWriter;
	outVideoWriter.open(videoname, CV_FOURCC('X', 'V', 'I', 'D'), 30, frames[0].size());
	for (unsigned int i = 0; i < frames.size(); i++)
	{
		printf("%04d\b\b\b\b", i);
		outVideoWriter << frames[i];
	}
	printf("\n");
	outVideoWriter.release();

}



int main(int argc,char *argv[])
{
	vector<cv::Mat>frames1, frames2, result;
	readvideo("leftsmooth.avi", frames1);
	readvideo("rightsmooth.avi", frames2);
	/*readvideo(argv[1], frames1);
	readvideo(argv[2], frames2);*/

	
	for (int i = 0; i < frames1.size(); i++)
	{
		cv::Mat left = frames1[i];
		cv::Mat right = frames2[i];
		cv::Mat resultIM = stitchgapIm(left, right);
		
		result.push_back(resultIM);
	}

	//writevideo(argv[3], result);
	writevideo("out8.avi", result);


	/*cv::Mat image0 = imread(".//left//k30.png");
	cv::Mat image1 = imread(".//right//k30.png");
	cv::Mat resultIM = stitchgapIm(image0, image1);*/
	
	
}

//int main(int argc, char *argv[])
//{
//	vector<cv::Mat>frames1,result;
//	readvideo("ok.avi", frames1);
//
//	int framestart = 5;
//	int frameend = 388;
//	for (int i = framestart; i < frameend; i++)
//	{
//		if (i==303)
//		{
//			continue;
//		}
//		cv::Mat left = frames1[i].clone();
//
//		result.push_back(left);
//	}
//	writevideo("output.avi", result);
//
//
//	/*cv::Mat image0 = imread(".//left//k30.png");
//	cv::Mat image1 = imread(".//right//k30.png");
//	cv::Mat resultIM = stitchgapIm(image0, image1);*/
//	
//	
//}

cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1)
{
	vector<cv::Mat>images_warp, mask, mask_graph;
	cv::Mat mask0, mask1;
	getImageNon_ZeroMask(image0, mask0);
	getImageNon_ZeroMask(image1, mask1);
	cv::Mat mask0_e, mask1_e;
	int subwidth = 25;
	int addwidthsubMask = 10;
	int addwidthMask = 10;
	cv::Mat mask0_di, mask1_di;

	erode(mask0, mask0_e, Mat(), Point(-1, -1), 20);
	erode(mask1, mask1_e, Mat(), Point(-1, -1), 20);
	/*dilate(mask0, mask0_di, Mat(addwidthMask, addwidthMask, CV_8U));
	dilate(mask1, mask1_di, Mat(addwidthMask, addwidthMask, CV_8U));*/



	cv::Mat allmask = mask0_e | mask1_e;
	cv::Mat graph_mask0 = allmask - mask1_e;
	//dilate(graph_mask0, mask0_di, Mat(),Point(-1,-1),25);
	//graph_mask0 = mask0_di&mask0_e;
	cv::Mat graph_mask1 = allmask - graph_mask0;



	imshow("graph_mask0.png", graph_mask0);
	imshow("graph_mask1.png", graph_mask1);

	/*imwrite("mask0.png", mask0);
	imwrite("mask1.png", mask1);*/



	vector<cv::Point>corners0, corners1;
	//findcorners(mask0, 4, corners0);////////////////////////////////////////�����ҵ���submask��corners
	//findcorners(mask1, 4, corners1);
	findboun_rect(mask0, corners0);
	findboun_rect(mask1, corners1);


	Rect rect0 = findRect(corners0);
	Rect rect1 = findRect(corners1);

	cv::Mat image0_ROI = image0(rect0);
	cv::Mat imag1_ROI = image1(rect1);
	cv::Mat graph_mask0_ROI = graph_mask0(rect0);
	cv::Mat graph_mask1_ROI = graph_mask1(rect1);
	cv::Mat mask0_ROI = mask0(rect0);
	cv::Mat mask1_ROI = mask1(rect1);

	images_warp.push_back(image0_ROI);
	images_warp.push_back(imag1_ROI);
	mask_graph.push_back(graph_mask0_ROI);
	mask_graph.push_back(graph_mask1_ROI);
	mask.push_back(mask0_ROI);
	mask.push_back(mask1_ROI);

	vector<Point>tls;
	tls.push_back(rect0.tl());
	tls.push_back(rect1.tl());

	cv::Point tl_result;
	tl_result.x = min(tls[0].x, tls[1].x);
	tl_result.y = min(tls[0].y, tls[1].y);

	cv::Mat result = ImageStitch(images_warp, mask, mask_graph, tls);

	cv::Mat result_final(image1.size(), CV_8UC3, Scalar::all(0));
	cv::Mat result_ROI = result_final(cv::Rect(tl_result.x, tl_result.y, result.cols, result.rows));
	result.copyTo(result_ROI);
	imshow("20", result_final);
	waitKey(20);

	return result_final;

}


cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1,int index)
{
	vector<cv::Mat>images_warp, mask, mask_graph;
	cv::Mat mask0, mask1;
	getImageNon_ZeroMask(image0, mask0);
	getImageNon_ZeroMask(image1, mask1);
	cv::Mat mask0_e, mask1_e;
	int subwidth = 25;
	int addwidthsubMask = 10;
	int addwidthMask = 10;
	cv::Mat mask0_di, mask1_di;

	/*erode(mask0, mask0_e, Mat(subwidth, subwidth, CV_8U));
	erode(mask1, mask1_e, Mat(subwidth, subwidth, CV_8U));
	dilate(mask0, mask0_di, Mat(addwidthMask, addwidthMask, CV_8U));
	dilate(mask1, mask1_di, Mat(addwidthMask, addwidthMask, CV_8U));*/

	cv::Mat allmask = mask0 | mask1;
	cv::Mat graph_mask0 = allmask - mask1;

	char stra[1024];
	sprintf(stra, ".//resource//test//%d.png", index);
	imwrite(stra, graph_mask0);

	dilate(graph_mask0, mask0_di, Mat(), Point(-1, -1), 10);

	char strb[1024];
	sprintf(strb, ".//resource//test2//%d.png", index);
	imwrite(strb, mask0_di);

	graph_mask0 = mask0_di&mask0;
	cv::Mat graph_mask1 = allmask - graph_mask0;


	char str1[1024];
	sprintf(str1, ".//resource//graph_mask0//%d.png", index);

	char str2[1024];
	sprintf(str2, ".//resource//graph_mask1//%d.png", index);

	char str3[1024];
	sprintf(str3, ".//resource//mask0//%d.png", index);

	char str4[1024];
	sprintf(str4, ".//resource//mask1//%d.png", index);

	imwrite(str1, graph_mask0);
	imwrite(str2, graph_mask1);

	imwrite(str3, mask0);
	imwrite(str4, mask1);



	vector<cv::Point>corners0, corners1;
	//findcorners(mask0, 4, corners0);////////////////////////////////////////�����ҵ���submask��corners
	//findcorners(mask1, 4, corners1);
	findboun_rect(mask0, corners0);
	findboun_rect(mask1, corners1);


	Rect rect0 = findRect(corners0);
	Rect rect1 = findRect(corners1);

	cv::Mat image0_ROI = image0(rect0);
	cv::Mat imag1_ROI = image1(rect1);
	cv::Mat graph_mask0_ROI = graph_mask0(rect0);
	cv::Mat graph_mask1_ROI = graph_mask1(rect1);
	cv::Mat mask0_ROI = mask0(rect0);
	cv::Mat mask1_ROI = mask1(rect1);

	images_warp.push_back(image0_ROI);
	images_warp.push_back(imag1_ROI);
	mask_graph.push_back(graph_mask0_ROI);
	mask_graph.push_back(graph_mask1_ROI);
	mask.push_back(mask0_ROI);
	mask.push_back(mask1_ROI);

	vector<Point>tls;
	tls.push_back(rect0.tl());
	tls.push_back(rect1.tl());

	cv::Point tl_result;
	tl_result.x = min(tls[0].x, tls[1].x);
	tl_result.y = min(tls[0].y, tls[1].y);

	cv::Mat result = ImageStitch(images_warp, mask, mask_graph, tls);

	cv::Mat result_final(image1.size(), CV_8UC3, Scalar::all(0));
	cv::Mat result_ROI = result_final(cv::Rect(tl_result.x, tl_result.y, result.cols, result.rows));
	result.copyTo(result_ROI);
	imshow("20", result_final);
	waitKey(20);

	return result_final;

}