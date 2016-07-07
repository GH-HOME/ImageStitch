/*******************************
*版本更新：2015、4、29
*******************************/


#include "CommonOperate.h"
#include <tchar.h>
#include <stdio.h>

using namespace std;
using namespace cv;

vector<string> getNameSets(string front_part, string back_part, int numSets)
{

	char namestr[1024];
	vector <string>namestrSet;

	for (int index(0); index < numSets; index++)
	{
		string name = front_part;
		sprintf(namestr, "%d", index);
		name += namestr;
		name += back_part;
		namestrSet.push_back(name);
		name.clear();
	}
	return namestrSet;


}

int addImagegap(cv::Mat&image, int gapx, int gapy, cv::Mat &image_gap)
{
	cv::Mat imagegap(2 * gapx + image.rows, 2 * gapy + image.cols, image.type(), cv::Scalar());
	cv::Mat imgpart1(imagegap, cv::Rect(gapx, gapy, image.cols, image.rows));
	image.copyTo(imgpart1); // copy image2 to image1 roi
	image_gap = imagegap.clone();
	return 1;

}

void Mat2Point(cv::Mat &H, cv::Point3d&points)
{
	points = (Point3d)H;
}

void Point2Mat(cv::Mat &H, cv::Point3d&points)
{
	H = cv::Mat(points);
}

void Mat2Vec(cv::Mat &H, vector<double>&H_vec)
{
	const int rows = H.rows;
	const int cols = H.cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			H_vec.push_back(H.at<double>(i, j));
		}

	}


}

void Mat2Vec(cv::Mat &H, vector<float>&H_vec)
{
	const int rows = H.rows;
	const int cols = H.cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			H_vec.push_back(float(H.at<double>(i, j)));
		}

	}


}

void transformPoint(cv::Mat &H, vector<cv::Point3f> &Inpointsets, vector<cv::Point3f> &Outpointsets, int gapx, int gapy)
{
	vector<float>H_vec;
	Mat2Vec(H, H_vec);
	vector<cv::Point3f>::iterator itin = Inpointsets.begin();
	for (; itin != Inpointsets.end(); itin++)
	{
		Point3f points;
		points.x = H_vec[0] * (*itin).x + H_vec[1] * (*itin).y + H_vec[2] * (*itin).z;
		points.y = H_vec[3] * (*itin).x + H_vec[4] * (*itin).y + H_vec[5] * (*itin).z;
		points.z = H_vec[6] * (*itin).x + H_vec[7] * (*itin).y + H_vec[8] * (*itin).z;
		points.x = points.x / points.z + gapx;
		points.y = points.y / points.z + gapy;
		Outpointsets.push_back(points);
	}


}


void transformPoint(cv::Mat &H, vector<cv::Point2f> &Inpointsets, vector<cv::Point2f> &Outpointsets, int gapx, int gapy)
{
	vector<float>H_vec;
	Mat2Vec(H, H_vec);
	for (unsigned int i = 0; i < H_vec.size(); i++)
	{
		cout << H_vec[i] << endl;
	}
	vector<cv::Point2f>::iterator itin = Inpointsets.begin();

	for (; itin != Inpointsets.end(); itin++)
	{
		Point2f points;
		points.x = H_vec[0] * (*itin).x + H_vec[1] * (*itin).y + H_vec[2];
		points.y = H_vec[3] * (*itin).x + H_vec[4] * (*itin).y + H_vec[5];
		float z = H_vec[6] * (*itin).x + H_vec[7] * (*itin).y + H_vec[8];
		points.x = points.x / z + gapx;
		points.y = points.y / z + gapy;
		Outpointsets.push_back(points);
	}


}

void myimshow(const cv::Mat &Img)
{
	cv::imshow("", Img);
	cv::namedWindow("");
	cv::waitKey(0);
}

void myWarpShow(cv::Mat &Img, cv::Mat &H)
{
	cv::Mat dst = cv::Mat::zeros(Img.rows, Img.cols, CV_8UC3);
	cv::warpPerspective(Img, dst, H, dst.size());
	myimshow(dst);
}


vector<string>  get_filelist(char *foldname)
{
	printf("start");
	vector<string> flist;
	printf("second");
	HANDLE file;
	WIN32_FIND_DATA fileData;
	char line[1024];
	wchar_t fn[1000];
	mbstowcs(fn, (const char*)foldname, 999);
	printf("%s\n", fn);
	file = FindFirstFile(fn, &fileData);
	FindNextFile(file, &fileData);
	while (FindNextFile(file, &fileData)){
		wcstombs(line, (const wchar_t*)fileData.cFileName, 259);
		flist.push_back(line);
	}
	return flist;
}

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void getDirectories(string path, vector<string>& files){
	long hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void initvectordim2(int rows, int cols, vector<vector<cv::Mat> > &ivec)
{
	ivec.resize(rows);//初始化行数
	for (int i = 0; i != rows; i++)
	{
		ivec[i].resize(cols);//每行初始化列数
	}
}

void initvectordim2_H(int rows, int cols, vector<vector<cv::Mat> > &ivec, cv::Mat H)
{
	ivec.resize(rows);//初始化行数
	for (int i = 0; i != rows; i++)
	{
		ivec[i].resize(cols);//每行初始化列数
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ivec[i][j] = H.clone();
		}
	}
}


void initvectordim1_H(int rows, vector<cv::Mat>&ivec, cv::Mat H)
{

	ivec.resize(rows);
	for (int i = 0; i < rows; i++)
	{
		cv::Mat H_temp = H.clone();
		ivec[i] = H_temp;
	}
}


void initvectordim2_o(int rows, int cols, vector<vector<cv::Point2f> > &ivec, cv::Point2f o)
{
	ivec.resize(rows);//初始化行数
	for (int i = 0; i != rows; i++)
	{
		ivec[i].resize(cols);//每行初始化列数
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ivec[i][j] = o;
		}
	}
}

string getFilenameFromePath(string FilePath, char file[])
{
	string Path = FilePath;
	int pos = Path.find_last_of('\\');
	string filename(Path.substr(pos + 1));
	strcpy(file, filename.c_str());
	return filename;
}

cv::Point2f matMyPointCVMat(const cv::Point2f &pt, const cv::Mat H){
	cv::Mat cvPt = cv::Mat::zeros(3, 1, CV_64F);
	cvPt.at<double>(0, 0) = pt.x;
	cvPt.at<double>(1, 0) = pt.y;
	cvPt.at<double>(2, 0) = 1.0;

	cv::Mat cvResult = H*cvPt;

	cv::Point2f result;
	result.x = cvResult.at<double>(0, 0) / cvResult.at<double>(2, 0);
	result.y = cvResult.at<double>(1, 0) / cvResult.at<double>(2, 0);

	return result;
}


int FindMax(vector<int> inputvec, int &index)
{
	int max = 0;
	for (int i = 0; i<inputvec.size(); i++)
	{
		if (inputvec[i]>max)
		{
			max = inputvec[i];
			index = i;
		}
	}
	return max;
}



int FindMax(vector<double> inputvec, int &index)
{
	double max = 0;
	for (int i = 0; i<inputvec.size(); i++)
	{
		if (inputvec[i]>max)
		{
			max = inputvec[i];
			index = i;
		}
	}
	return max;
}



double Average(vector<int> inputvec)
{
	double sum = 0;
	for (int i = 0; i < inputvec.size(); i++)
	{
		sum += inputvec[i];
	}
	double average = sum / inputvec.size();
	return average;
}

/*交换函数，作用是交换数组中的两个元素的位置*/
void swap(int array[], int i, int j)
{
	int tmp = array[i];
	array[i] = array[j];
	array[j] = tmp;
}

void swap(vector<int> &ivec, int i, int j)
{
	int tmp = ivec[i];
	ivec[i] = ivec[j];
	ivec[j] = tmp;
}


void swap(vector<double> &ivec, int i, int j)
{
	double tmp = ivec[i];
	ivec[i] = ivec[j];
	ivec[j] = tmp;
}

/*选择排序*/
void SelectionSort(int array[], int n)
{
	for (int i = 0; i < n - 1; i++)
	{
		int smallest = i;
		for (int j = i + 1; j<n; j++)
		{
			if (array[smallest]>array[j])
				smallest = j;
		}
		swap(array, i, smallest);
	}
}

void SelectionSort(vector <int>&ivec, vector <int>&Sortindex)
{
	int n = ivec.size();
	vector<int>ivec_save = ivec;
	Sortindex.resize(n);
	for (int i = 0; i < n - 1; i++)
	{
		int max = i;
		for (int j = i + 1; j < n; j++)
		{
			if (ivec[max] < ivec[j])
			{
				max = j;
				Sortindex[i] = j;
			}

		}
		swap(ivec, i, max);
	}
	for (int p = 0; p < ivec.size(); p++)
	{
		for (int q = 0; q < ivec_save.size(); q++)
		{
			if (ivec[p] == ivec_save[q])
			{
				Sortindex[p] = q;
			}
		}
	}
}

//flag=0 DECREASE  flag=1 INCREASE
void SelectionSort(vector <double>&ivec, vector <int>&Sortindex, int flag)
{
	int n = ivec.size();
	vector<double>ivec_save = ivec;
	Sortindex.resize(n);
	for (int i = 0; i < n - 1; i++)
	{
		double max = i;
		for (int j = i + 1; j < n; j++)
		{
			if (flag == 0)
			{
				if (ivec[max] < ivec[j])
				{
					max = j;
				}
			}
			else if (flag == 1)
			{
				if (ivec[max] > ivec[j])
				{
					max = j;
				}
			}


		}
		swap(ivec, i, max);
	}
	for (int p = 0; p < ivec.size(); p++)
	{
		for (int q = 0; q < ivec_save.size(); q++)
		{
			if (ivec[p] == ivec_save[q])
			{
				Sortindex[p] = q;
			}
		}
	}
}

int findmaxarea(vector<int> inputvec, int windowsize)
{
	int maxnum = 3;
	vector<int>sortvec;
	sortvec = inputvec;
	vector<int> Sortindex;
	Sortindex.resize(inputvec.size());
	SelectionSort(sortvec, Sortindex);

	vector<double>avg_vec(maxnum);
	for (int i = 0; i < maxnum; i++)
	{
		double max = Sortindex[i];
		int numadd = 0;
		double sum = 0;
		for (int j = -windowsize; j < windowsize; j++)
		{
			if ((max + j >= 0) && (max + j < inputvec.size()))
			{
				sum += inputvec[max + j];
				numadd++;
			}
		}
		double average = sum / numadd;
		avg_vec[i] = average;
	}
	int Idx = 0;
	FindMax(avg_vec, Idx);
	int resultIdx = Sortindex[Idx];
	return resultIdx;
}

int findmaxarea(vector<double> inputvec, int windowsize)
{
	int maxnum = 3;
	vector<double>sortvec;
	sortvec = inputvec;
	vector<int> Sortindex;
	Sortindex.resize(inputvec.size());
	SelectionSort(sortvec, Sortindex, 1);

	vector<double>avg_vec(maxnum);
	for (int i = 0; i < maxnum; i++)
	{
		double max = Sortindex[i];
		int numadd = 0;
		double sum = 0;
		for (int j = -windowsize; j < windowsize; j++)
		{
			if ((max + j >= 0) && (max + j < inputvec.size()))
			{
				sum += inputvec[max + j];
				numadd++;
			}
		}
		double average = sum / numadd;
		avg_vec[i] = average;
	}
	int Idx = 0;
	FindMax(avg_vec, Idx);
	int resultIdx = Sortindex[Idx];
	return resultIdx;
}


void MyDrawPolygon(Mat &img, vector<cv::Point2f>points)
{
	int lineType = 8;

	Point*PointSet = new Point[points.size()];

	for (int i = 0; i < points.size(); i++)
	{
		PointSet[i] = (Point)points[i];
	}


	const Point* ppt[1] = { PointSet };
	int npt[] = { points.size() };

	fillPoly(img,
		ppt,
		npt,
		1,
		Scalar(255, 255, 255),
		lineType);
}


/****************************************************************************************************
*返回image图像的非0部分的mask，事实上实际拍摄中的照片不管怎么样像素不可能纯为0，为0的都是人为加的gap
****************************************************************************************************/
void getImageNon_ZeroMask(cv::Mat image, cv::Mat &mask)
{
	cv::Mat image_gray(image.size(), CV_8UC1);
	if (image.channels() == 3)
	{
		cvtColor(image, image_gray, CV_BGR2GRAY);
	}
	else if (image.channels() == 1)
	{
		image_gray = image.clone();
	}
	cv::Mat maskresult;
	maskresult.create(image_gray.size(), CV_8UC1);
	maskresult.setTo(Scalar::all(0));
	for (int i = 0; i < image_gray.rows; i++)
	{
		for (int j = 0; j < image_gray.cols; j++)
		{
			if (image_gray.at<uchar>(i, j) > 1)
			{
				maskresult.at<uchar>(i, j) = 255;
			}
			else
			{
				maskresult.at<uchar>(i, j) = 0;
			}

		}
	}
	
	cv::Mat mask_erode, mask_dilate;
	
	erode(maskresult, mask_erode, Mat(), Point(-1, -1), 5); //这里是为了去除mask周边的一些噪声
	

	dilate(mask_erode, mask_dilate, Mat(), Point(-1, -1), 30);
	erode(mask_dilate, mask_erode, Mat(), Point(-1, -1), 30);

	mask = mask_erode.clone();
}

void getImageNon_ZeroMask(cv::Mat image, cv::Mat &mask,int value)
{
	cv::Mat image_gray(image.size(), CV_8UC1);
	if (image.channels() == 3)
	{
		cvtColor(image, image_gray, CV_BGR2GRAY);
	}
	else if (image.channels() == 1)
	{
		image_gray = image.clone();
	}
	cv::Mat maskresult;
	maskresult.create(image_gray.size(), CV_8UC1);
	maskresult.setTo(Scalar::all(0));
	for (int i = 0; i < image_gray.rows; i++)
	{
		for (int j = 0; j < image_gray.cols; j++)
		{
			if (image_gray.at<uchar>(i, j) != 0)
			{
				maskresult.at<uchar>(i, j) = value;
			}

		}
	}

	mask = maskresult.clone();
}

bool isPointinLine(cv::Point2f pt, cv::Point2f V1, cv::Point2f V2, double threshold)
{
	double A1 = (pt.y - V1.y) / (pt.x - V1.x);
	double A2 = (pt.y - V1.y) / (pt.x - V1.x);
	if (abs(A1 - A2) <= threshold)
	{
		return true;
	}
	else
		return false;
}


void getMaskShapeImage(cv::Mat SrcIm, cv::Mat &DstIm, cv::Mat mask)
{
	cv::Mat image(SrcIm.size(), CV_8UC3);
	image.setTo(Scalar::all(0));
	for (int i = 0; i < SrcIm.rows; i++)
	{
		for (int j = 0; j < SrcIm.cols; j++)
		{
			if (mask.at<uchar>(i, j) != 0)
			{
				image.at<Vec3b>(i, j) = SrcIm.at<Vec3b>(i, j);
			}
		}
	}
	image.copyTo(DstIm);
}

void drawfeatures(cv::Mat &image, vector<cv::Point>features)
{
	cv::Mat temp = image.clone();
	cv::Scalar color2(0, 255, 0);
	int gap = 0;
	int lineWidth = 2;
	for (int i = 0; i < features.size(); i++)
	{
		cv::circle(temp, cv::Point(features[i].x, features[i].y), lineWidth + 2, color2, -1);
	}
	temp.copyTo(image);
}

void removeRepeatPts(vector<cv::Point2f>quaryPts, vector<cv::Point2f>&trainPts)
{
	vector<cv::Point2f>dependentPts;
	dependentPts.resize(0);
	for (int i = 0; i < trainPts.size(); i++)
	{
		if (!isPointInVec(trainPts[i], quaryPts))
		{
			dependentPts.push_back(trainPts[i]);
		}
	}
	trainPts.resize(0);
	trainPts = dependentPts;
}

bool isPointInVec(cv::Point2f pt, vector<cv::Point2f>vec)
{

	for (int i = 0; i < vec.size(); i++)
	{
		if (pt == vec[i])
		{
			return true;
		}
	}
	return false;
}

bool removeRepeatPtsInoneVec(vector<cv::Point2f>&quaryPts, vector<cv::Point2f>&trainPts)
{
	//quaryPts和trainPts是一一对应的两组点，但是quaryPts中有一些重复的点，我们去除这些重复的点，并且把这些重复的点在trainPts对应的点也去掉
	if (quaryPts.size() != trainPts.size())
	{
		cerr << "quaryPts size not match trainPts" << endl;
		return false;
	}
	vector<cv::Point2f> temp1, temp2;
	temp1.resize(quaryPts.size());
	temp2.resize(trainPts.size());
	for (int i = 0; i < quaryPts.size(); i++){
		temp1[i] = quaryPts[i];
		temp2[i] = trainPts[i];
	}
	quaryPts.clear();
	trainPts.clear();
	for (int i = 0; i < temp1.size(); i++)
	{
		if (!isPointInVec(temp1[i], quaryPts))
		{
			quaryPts.push_back(temp1[i]);
			trainPts.push_back(temp2[i]);
		}
	}
	return true;

}

void findcorners(cv::Mat image, int num_corners,vector<cv::Point>&corners)
{
	// 改进的harris角点检测方法
	goodFeaturesToTrack(image, corners,
		num_corners,
		//角点最大数目
		0.05,
		// 质量等级，这里是0.01*max（min（e1，e2）），e1，e2是harris矩阵的特征值
		25);
	// 两个角点之间的距离容忍度

}
Rect findRect(vector<cv::Point>corners)
{
	Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
	Point br(numeric_limits<int>::min(), numeric_limits<int>::min());
	for (size_t i = 0; i < corners.size(); ++i)
	{
		tl.x = min(tl.x, corners[i].x);
		tl.y = min(tl.y, corners[i].y);
		br.x = max(br.x, corners[i].x);
		br.y = max(br.y, corners[i].y);
	}

	return Rect(tl, br);
}


cv::Mat circleMatrix(int radius)
{
	int width = 2 * radius;
	int height = 2 * radius;
	cv::Mat mask(width, height, CV_8UC1, cv::Scalar(0));
	cv::Point2f center(radius+0.5, radius+0.5);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cv::Point2f pt(i+1, j+1);
			double temp = norm(center - pt);

			if (norm(center-pt)<radius)
			{
				mask.at<uchar>(j, i) = 1;
			}
			else
			{
				mask.at<uchar>(j, i) = 0;
			}
		}
	}
	return mask;
}

void findboun_rect(cv::Mat mask, vector<cv::Point>&corners)
{
	int height = mask.rows;
	int width = mask.cols;

	int top = 0, bottom = 0;
	int left = 0,right = 0;

	bool flag_t = false;
	bool flag_b = false;
	bool flag_l = false;
	bool flag_r = false;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (mask.at<uchar>(i,j)==255&&flag_t==false)
			{
				top = i;
				flag_t = true;
			}
			if (mask.at<uchar>(height-i-1,j)==255&&flag_b==false)
			{
				bottom = height - i - 1;
				flag_b = true;
			}
			if (flag_t&&flag_b)
			{
				break;
			}
		}
	}

	for (int j = 0; j < width; j++)
	{
		for (int i = 0; i < height; i++)
		{
		
			if (mask.at<uchar>(i, j) == 255 && flag_l == false)
			{
				left = j;
				flag_l = true;
			}
			if (mask.at<uchar>(i, width - j - 1) == 255 && flag_r == false)
			{
				right = width - j - 1;
				flag_r = true;
			}
			if (flag_l&&flag_r)
			{
				break;
			}
		}
	}

	corners.resize(4);
	corners[0] = cv::Point(left, top);
	corners[1] = cv::Point(right, top);
	corners[2] = cv::Point(left, bottom);
	corners[3] = cv::Point(right, bottom);



	/*cout << "left  " << left << "right   " << right << endl;
	cout << "top  " << top << "bottom   " << bottom << endl;
	system("pause");*/
}



cv::Mat calcgradient(const cv::Mat& img)
{
	cv::Mat image_gray;
	cv::Mat result(img.size(), CV_64FC1);
	if (img.channels() != 1)
	{
		cvtColor(img, image_gray, CV_BGR2GRAY);
	}
	else
	{
		image_gray = img;
	}
	
	image_gray.convertTo(image_gray, CV_64FC1);
	double tmp = 0;
	int rows = image_gray.rows - 1;
	int cols = image_gray.cols - 1;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double dx = image_gray.at<double>(i, j + 1) - image_gray.at<double>(i, j);
			double dy = image_gray.at<double>(i + 1, j) - image_gray.at<double>(i, j);
			double ds = std::sqrt((dx*dx + dy*dy)/2);
			result.at<double>(i, j) = ds;
		}
	}
	
	return result;
}




double addROIPix(cv::Mat image, cv::Mat mask)
{
	cv::Mat gray;
	int sum_nonzero_Pix=0;

	if (image.channels()!=1)
	{
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	
	else
	{
		gray = image.clone();
	}
	
	assert(gray.size == mask.size);
	double sum = 0.0;


	vector<double>gradient_vec;
	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{

			if (mask.at<uchar>(i,j)==255)
			{
				gradient_vec.push_back(abs(gray.at<double>(i, j)));
				
			}
		}
	}

	sort(gradient_vec.begin(),gradient_vec.end());

	int num = gradient_vec.size();

	for (int i = gradient_vec.size(); i > gradient_vec.size()-num; i--)
	{
		sum += gradient_vec[i-1];
		sum_nonzero_Pix++;
	}


	return sum / num;
}


cv::Mat getMaskcontour(cv::Mat mask,int width)
{
	cv::Mat mask_small,mask_big;
	erode(mask, mask_small, Mat(), Point(-1, -1), width+3);
	erode(mask, mask_big, Mat(), Point(-1, -1), width+2);

	cv::Mat result = mask_big - mask_small;
	return result;

}


void SubMatrix(cv::Mat matrix1, cv::Mat matrix2,cv::Mat& result)
{
	cv::Mat temp(matrix1.size(), CV_64FC1);
	for (int i = 0; i < matrix1.rows;i++)
	{
		for (int j = 0; j < matrix1.rows;j++)
		{
			temp.at<double>(i, j) = matrix1.at<double>(i, j) - matrix2.at<double>(i, j);
			
			/*if (temp.at<double>(i, j)>100)
			{
				cout << " "<<i  << "  "<< j << endl;
				system("pause");
			}*/
		
		}
	}
	result = temp.clone();


}