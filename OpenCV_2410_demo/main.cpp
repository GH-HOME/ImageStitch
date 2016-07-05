#include "User_System.h"
#include "CommonOperate.h"
#include "MyDebug.h"
#include "save.h"
#include "MyImageStitch.h"
#include "RobustMatcher.h"

using namespace std;
using namespace cv;

cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1, int index);

cv::Mat stitchgapIm(cv::Mat image0, cv::Mat image1);

double mysum = 0.0;
double mymax = 0.0;
int my_count = 0;
double max_gradient_all = 0.0;
double min_gradient_all = 10000000;


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

void image_crop(int x_l, int x_r, int y_u, int y_d, cv::Mat source, cv::Mat &target)
{
	cv::Mat temp = source.clone();
	int width = source.cols - x_l - x_r;
	int height = source.rows - y_u - y_d;
	cv::Mat imageROI = temp(Rect(x_l, y_u, width, height));
	imageROI.copyTo(target);
}

//int main(int argc,char *argv[])
//{
//	vector<cv::Mat>frames1, frames2, result;
//
//	readvideo("fly.avi", frames1);
//	writevideo("output.avi", frames1);
//
//
//
//
//	/*cv::Mat image0 = imread(".//left//k30.png");
//	cv::Mat image1 = imread(".//right//k30.png");
//	cv::Mat resultIM = stitchgapIm(image0, image1);*/
//	
//	
//}


//int main(int argc,char *argv[])
//{
//	vector<cv::Mat>frames1, frames2;
//
//	readvideo("output4.avi", frames1);
//
//	for (int i = 0; i < frames1.size();i++)
//	{
//		cv::Mat I = frames1[i];
//		cv::Mat result;
//		image_crop(384, 476, 533, 428, I, result);
//		frames2.push_back(result);
//		
//	}
//	writevideo("output.avi", frames2);
//
//
//
//
//	/*cv::Mat image0 = imread(".//left//k30.png");
//	cv::Mat image1 = imread(".//right//k30.png");
//	cv::Mat resultIM = stitchgapIm(image0, image1);*/
//	
//	
//}




cv::Mat blending(cv::Mat image0, cv::Mat image1)
{
	vector<cv::Mat>images_warp, mask, mask_graph;
	cv::Mat mask0, mask1;
	getImageNon_ZeroMask(image0, mask0);
	getImageNon_ZeroMask(image1, mask1);

	cv::Mat result(mask1.size(), CV_8UC3);
	for (int i = 0; i < mask0.rows; i++)
	{
		for (int j = 0; j < mask0.cols; j++)
		{
			//result.at<Vec3b>(i, j) = image1.at<Vec3b>(i, j);
			if (mask0.at<uchar>(i, j) && mask1.at<uchar>(i, j))
			{
				result.at<Vec3b>(i, j)[0] = (image0.at<Vec3b>(i, j)[0] + image1.at<Vec3b>(i, j)[0]) / 2;
				result.at<Vec3b>(i, j)[1] = (image0.at<Vec3b>(i, j)[1] + image1.at<Vec3b>(i, j)[1]) / 2;
				result.at<Vec3b>(i, j)[2] = (image0.at<Vec3b>(i, j)[2] + image1.at<Vec3b>(i, j)[2]) / 2;
			}
			else if (mask0.at<uchar>(i, j))
			{
				result.at<Vec3b>(i, j) = image0.at<Vec3b>(i, j);
			}
			else
			{
				result.at<Vec3b>(i, j) = image1.at<Vec3b>(i, j);
			}
		}
	}
	return result;
}

#if 1
int main(int argc, char *argv[])
{
	
	string videoname = "output6.avi";
	string videoname3 = ".//8//left.avi";
	string videoname4 = ".//8//right.avi";
	string videoname1 = ".//8//outleft.avi";
	string videoname2 = ".//8//outright.avi";
	printf("Read Video\n");
	cout << "now read video..." << videoname1 << endl;
	cv::VideoCapture capture(videoname1);
	// check if video successfully opened
	if (!capture.isOpened())
	{
		cerr << "The video can not open!";
		return -1;
		exit(0);
	}

	
	printf("Read Video\n");
	cout << "now read video" << videoname2 << endl;
	cv::VideoCapture capture2(videoname2);

	cv::VideoCapture capture3(videoname3);
	cv::VideoCapture capture4(videoname4);
	int video_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int video_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	// check if video successfully opened
	if (!capture2.isOpened())
	{
		cerr << "The video can not open!";
		return -1;
		exit(0);
	}

	cv::VideoWriter outVideoWriter;
	outVideoWriter.open(videoname, CV_FOURCC('X', 'V', 'I', 'D'), 30, cv::Size(video_width+2*40, video_height+2*40));

	cv::Mat frame1,frame2,frame3,frame4; // current video frame
	printf("extract frames...");
	int frame_count = 0;

	while (capture3.read(frame3)&& capture4.read(frame4))
	{

		
		printf("%04d\b\b\b\b", frame_count);
		frame_count++;

		//cout << "frame index    " << frame_count << endl;

		cv::Mat gray_image = calcgradient(frame3);

		cv::Mat gray_image2 = calcgradient(frame4);

		Scalar sum_temp1 = sum(gray_image2);
		double sum_gradient = sum_temp1.val[0];

		Scalar sum_temp = sum(gray_image);
		double sum_gradient2 = sum_temp.val[0];

		if (sum_gradient>max_gradient_all)
		{
			max_gradient_all = sum_gradient;
		}
		if (sum_gradient2>max_gradient_all)
		{
			max_gradient_all = sum_gradient2;
		}

		if (sum_gradient < min_gradient_all)
		{
			min_gradient_all = sum_gradient;
		}
		if (sum_gradient2 < min_gradient_all)
		{
			min_gradient_all = sum_gradient2;
		}
		
	}

	
	min_gradient_all = min_gradient_all / (640.0 * 360.0);
	max_gradient_all /= (640.0 * 360.0);
	//min_gradient_all = max_gradient_all - min_gradient_all;
	//min_gradient_all = 0.0;

	cout<<"max_gradient_all  " << max_gradient_all << endl;
	cout <<"min_gradient_all  "<< min_gradient_all << endl;
	
	frame_count = 0;
	while (capture.read(frame1) && capture2.read(frame2))
	{
		cv::Mat frame_copy1 = frame1.clone();
		cv::Mat frame_copy2 = frame2.clone();

		cv::Mat image1_gap, image2_gap;
		addImagegap(frame_copy1, 40, 40, image1_gap);
		addImagegap(frame_copy2, 40, 40, image2_gap);

		if (frame_count>93 && frame_count<293)
		{
			cv::Mat resultIM = stitchgapIm(image1_gap, image2_gap);
		}
		
		//outVideoWriter << resultIM;
		printf("%04d\b\b\b\b", frame_count);
		frame_count++;
		

		
		printf("max value is %lf\n", mymax);
		
	}


	system("pause");

	capture.release();
	capture2.release();
	outVideoWriter.release();


}

#endif



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


	//method1
	erode(mask0, mask0_e, Mat(), Point(-1, -1), 20);
	erode(mask1, mask1_e, Mat(), Point(-1, -1), 20);
	cv::Mat allmask = mask0_e | mask1_e;
	cv::Mat graph_mask0 = allmask - mask1_e;
	cv::Mat graph_mask1 = allmask - graph_mask0;


	//method2
	/*erode(mask0, mask0, Mat(), Point(-1, -1), 2);
	erode(mask1, mask1, Mat(), Point(-1, -1), 2);

	cv::Mat allmask = mask0 | mask1;
	cv::Mat graph_mask0 = allmask - mask1;
	dilate(graph_mask0, mask0_di, Mat(),Point(-1,-1),30);
	graph_mask0 = mask0_di&mask0;
	cv::Mat graph_mask1 = allmask - graph_mask0;*/



	vector<cv::Point>corners0, corners1;
	//findcorners(mask0, 4, corners0);////////////////////////////////////////这里找的是submask的corners
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
	/*imshow("20", result_final);
	waitKey(20);*/


	cv::Mat overlapmask = mask0_e&mask1_e;


	//cv::Mat image11;
	//getMaskShapeImage(image1, image11, graph_mask1);
	
	cv::Mat gray_image = calcgradient(image1);


	cv::Mat gray_result = calcgradient(result_final);
	cv::Mat mask_contour = getMaskcontour(overlapmask, 1);


	gray_result = (gray_result - min_gradient_all) / (max_gradient_all - min_gradient_all) * 10;
	gray_image = (gray_image - min_gradient_all) / (max_gradient_all - min_gradient_all) * 10;


	cv::Mat mask_shape1, mask_shape2;
	getMaskShapeImage(image0, mask_shape1, mask_contour);
	getMaskShapeImage(result_final, mask_shape2, mask_contour);
	
	imwrite("mask_shape2.png", mask_shape2);
	imwrite("mask_shape1.png", mask_shape1);
	imwrite("image0.png", image0); imwrite("result_final.png", result_final);


	/*cv::Mat sub_image;
	
	SubMatrix(gray_result, gray_image, sub_image);*/

	//double now_value = addROIPix(sub_image, mask_contour);
	double now_value = subROIPix(gray_image, gray_result, mask_contour);

	
	//cout << "orgin_value     " << orgin_value << endl;
	if (mymax<now_value)
	{
		/*if (mymax>200)
		{
			my_count++;
			cout << "bad   " << my_count << endl;
		}*/
		//else
		{
			mymax = now_value;
		}
		
	}


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
	//findcorners(mask0, 4, corners0);////////////////////////////////////////这里找的是submask的corners
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


