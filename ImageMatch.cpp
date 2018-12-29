
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;


int main()
{
	// 打开影像
	Mat image01 = imread("img505.jpg", 1);
	Mat image02 = imread("img507.jpg", 1);
	namedWindow("左片", 0);
	imshow("左片", image01);
	namedWindow("右片", 0);
	imshow("右片", image02);

	//灰度图转换  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);


	//SIFT算子提取特征点    
	SiftFeatureDetector siftDetector(800);  
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image1, keyPoint1);
	siftDetector.detect(image2, keyPoint2);

	//特征点描述，为下边的特征点匹配做准备    
	SiftDescriptorExtractor SiftDescriptor;
	Mat imageDesc1, imageDesc2;
	SiftDescriptor.compute(image1, keyPoint1, imageDesc1);
	SiftDescriptor.compute(image2, keyPoint2, imageDesc2);

	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;

	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();

	matcher.knnMatch(imageDesc2, matchePoints, 2);
	cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}

	Mat first_match;
	drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
	namedWindow("match ", 0);
	imshow("match ", first_match);
	imwrite("match.jpg", first_match);
	waitKey();
	return 0;
}