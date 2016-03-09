#include "MyDebug.h"

using namespace std;
using namespace cv;


void MyPrint(cv::Mat &H){
	for (int i = 0; i < H.rows; i++){
		for (int j = 0; j < H.cols; j++){
			printf("%f ", H.at<double>(i, j));
		}
		printf("\n");
	}
}


void MyPrint(vector<cv::Mat> &HList){
	vector<cv::Mat>::iterator it = HList.begin();
	int index = 0;
	for (; it != HList.end(); it++)
	{
		cout << "index [" << index << "]" << endl;
		MyPrint(*it);
		cout << endl;
		index++;
	}

}


int MatrixShow(const cv::Mat &matrix, char* fileName)
{
	if (matrix.empty())
	{
		cerr << "The matrix is empty!";
		return -1;
	}

	FILE* fd = fopen(fileName, "w");

	if (fd == NULL)
	{
		cerr << "The file can not be opened";
		return -1;
	}
	int height = matrix.rows;
	int width = matrix.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (matrix.channels() == 1)
			{
				printf("%d ", matrix.at<uchar>(i, j));         //  gray value 0~250
				fprintf(fd, "%d ", matrix.at<uchar>(i, j));         //  gray value 0~250
			}

			else if (matrix.channels() == 3)
			{
				printf("[ ");
				printf("%d ", matrix.at<cv::Vec3b>(i, j)[0]); //b 0~255
				printf("%d ", matrix.at<cv::Vec3b>(i, j)[1]); //g 0~255
				printf("%d ", matrix.at<cv::Vec3b>(i, j)[2]); //r 0~255
				printf(" ]");

				fprintf(fd, "[ ");
				fprintf(fd, "%d ", matrix.at<cv::Vec3b>(i, j)[0]); //b 0~255
				fprintf(fd, "%d ", matrix.at<cv::Vec3b>(i, j)[1]); //g 0~255
				fprintf(fd, "%d ", matrix.at<cv::Vec3b>(i, j)[2]); //r 0~255
				fprintf(fd, " ]");
			}


		}
		printf("\n");
		fprintf(fd, "\n");
	}
	fclose(fd);

	return 0;
}


int MatrixShow(vector<cv::Mat> &matrixList, char* fileName)
{
	if (matrixList.empty())
	{
		cerr << "The matrix is empty!";
		return -1;
	}


	FILE* fd = fopen(fileName, "w");

	if (fd == NULL)
	{
		cerr << "The file can not be opened";
		return -1;
	}

	vector<cv::Mat>::iterator it = matrixList.begin();
	int index = 0;
	for (; it != matrixList.end(); it++)
	{

		vector<cv::Mat>::iterator it = matrixList.begin();
		int height = (*it).rows;
		int width = (*it).cols;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*it).channels() == 1)
				{
					printf("%d ", (*it).at<uchar>(i, j));         //  gray value 0~250
					fprintf(fd, "%d ", (*it).at<uchar>(i, j));         //  gray value 0~250
				}

				else if ((*it).channels() == 3)
				{
					printf("[ ");
					printf("%d ", (*it).at<cv::Vec3b>(i, j)[0]); //b 0~255
					printf("%d ", (*it).at<cv::Vec3b>(i, j)[1]); //g 0~255
					printf("%d ", (*it).at<cv::Vec3b>(i, j)[2]); //r 0~255
					printf(" ]");

					fprintf(fd, "[ ");
					fprintf(fd, "%d ", (*it).at<cv::Vec3b>(i, j)[0]); //b 0~255
					fprintf(fd, "%d ", (*it).at<cv::Vec3b>(i, j)[1]); //g 0~255
					fprintf(fd, "%d ", (*it).at<cv::Vec3b>(i, j)[2]); //r 0~255
					fprintf(fd, " ]");
				}


			}
			printf("\n");
			fprintf(fd, "\n");
		}
	}
	fclose(fd);

	return 0;
}


//int MatrixShow(vector<cv::Mat> &matrixList, char* fileName)
//{
//	if (matrixList.empty())
//	{
//		cerr << "The matrix is empty!";
//		return -1;
//	}
//
//
//	FILE* fd = fopen(fileName, "w");
//
//	if (fd == NULL)
//	{
//		cerr << "The file can not be opened";
//		return -1;
//	}
//
//	vector<cv::Mat>::iterator it = matrixList.begin();
//	int index = 0;
//	for (; it != matrixList.end(); it++)
//	{
//
//		vector<cv::Mat>::iterator it = matrixList.begin();
//
//	cv:Mat_ <double>::iterator itM = (*it).begin<double>();
//		for (; itM != (*it).end<double>(); it++)
//		{
//			MyPrint(*it);
//			printf("%d ", (*it));         //  gray value 0~250
//		}
//	
//	
//	
//	}
//
//	return 0;
//}
