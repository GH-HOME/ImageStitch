#include "save.h"
#include "MyDebug.h"
using namespace cv;
using namespace std;


int VectorSave(vector<cv::Mat> &matrix, char*path)
{
	vector<cv::Mat>::iterator it = matrix.begin();
	std::ofstream FILE(path, std::ios::out |std::ios::app|std::ofstream::binary);
	for (; it != matrix.end(); it++)
	{
		/*cout << *it;
		cout << endl;*/
		FILE << *it;
		FILE << endl;
	}
	return 0;
}


int data2file(vector<cv::Mat> &matrix, char*path)
{
	ofstream out_file(path);
	if (!out_file)
	{
		cout << "files open failed!\n";
		return -1;
	}
	ostream_iterator< cv::Mat > os(out_file, " ");
	copy(matrix.begin(), matrix.end(), os);
	return 0;

}

int PointSave(vector<cv::Point2f> &SrcPoints, vector<cv::Point2f> &DstPoints, string path)
{
	vector<cv::Point2f>::iterator its = SrcPoints.begin();
	vector<cv::Point2f>::iterator itd = DstPoints.begin();
	std::ofstream FILE(path, std::ios::out | std::ofstream::binary);
	for (; its != SrcPoints.end(); its++, itd++)
	{

		FILE << (*its).x<<" ";
		FILE << (*its).y << " ";



		FILE << (*itd).x << " ";
		FILE << (*itd).y << " ";


		FILE << endl;
	}
	return 0;
}

//int readHlist_txt(char*filename, vector<cv::Mat>&HomoList, int numFrames)
//{
//	int H_arr[3][3]; /*假定不超过10行,每行一定有4个元素*/
//	vector<cv::Mat>HList;
//	int i, j;
//	FILE *fp;
//	/*打开文件*/
//	fp = fopen(filename, "r");  /*假设a.txt在d盘根目录下*/
//	if (!fp)exit(0);
//	for (int H_num = 0; H_num < numFrames - 1; H_num++)
//	{
//		for (j = H_num * 4; j < j + 3; j++) /*假定有j行*/
//		{
//			for (i = 0; i < 3; i++)
//			{
//				fscanf(fp, "%d", &H_arr[j][i]);  /*读一个数据*/
//				printf("i=%d,j=%d H_arr[j][i]=%f\n", i, j, H_arr[j][i]);
//
//			}
//
//		}
//		cv::Mat H(3, 3, CV_64F, H_arr);
//		cout << H;
//		HList.push_back(H);
//	}
//	HomoList = HList;
//	/* 关闭文件 */
//	fclose(fp);
//
//	return 0;
//}


int readPointSet_txt(string filename, vector<cv::Point2f>&SrcPoints,vector<cv::Point2f>&DetPoints)
{
	cv::Point2f sourcepoint,destpoint;
	char buffer[1024];
	ifstream myfile(filename);
	while (!myfile.eof())
	{
		myfile.getline(buffer, 100);
		sscanf(buffer, "%f %f %f %f", &sourcepoint.x, &sourcepoint.y, &destpoint.x, &destpoint.y);
		//cout << sourcepoint.x << " " << sourcepoint.y << " "<<destpoint.x << " " << destpoint.y << endl;
		SrcPoints.push_back(sourcepoint);
		DetPoints.push_back(destpoint);
	}
	myfile.close();
	return 0;
}

int readHlist_txt(char*filename, vector<cv::Mat>&HomoList, int numFrames)
{
	int H_arr[3][3]; /*假定不超过10行,每行一定有4个元素*/
	vector<cv::Mat>HList;
	HList.resize(numFrames);
	char buffer0[1024];
	char buffer1[1024];
	char buffer2[1024];
	
	ifstream myfile(filename);
	for (int i = 0; i < numFrames;i++)
	{
		cv::Mat H(3, 3, CV_64F);
		myfile.getline(buffer0, 100);
		sscanf(buffer0, "%lf %lf %lf", &H.at<double>(0, 0), &H.at<double>(0, 1), &H.at<double>(0, 2));
		//cout << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << " " << endl;
		myfile.getline(buffer1, 100);
		sscanf(buffer1, "%lf %lf %lf", &H.at<double>(1, 0), &H.at<double>(1, 1), &H.at<double>(1, 2));
		//cout << H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << " " << endl;

		myfile.getline(buffer2, 100);
		sscanf(buffer2, "%lf %lf %lf", &H.at<double>(2, 0), &H.at<double>(2, 1), &H.at<double>(2, 2));
		//cout << H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2) << " " << endl;

		//cout << H << endl;
		HList[i] = H;
		
	}
	//MyPrint(HList);
	HomoList = HList;
	//MyPrint(HomoList);
	myfile.close();
	return 0;
}


