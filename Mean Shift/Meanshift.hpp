#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

class PixelParameters
{
public:
	int x{ 0 };	// Image x-coordinate 
	int y{ 0 };	// Image y-coordinate
	float l{ 0 };	// L Channel 
	float a{ 0 };	// A Channel 
	float b{ 0 };	// B Channel 

public:
	PixelParameters();											
	void PointLab();											
	void PointRGB();											
	void AccumulatePoints(PixelParameters);								
	void CopyPoint(PixelParameters);								
	float ColorDistance(PixelParameters);						
	float SpatialDistance(PixelParameters);					
	void PointScale(float);									
	void SetPointValue(float, float, float, float, float);		
};

class MeanShift
{
public:
	float SpatialRadius;
	float ColorRadius;	
	vector<Mat> IMGChannels;

public:
	MeanShift(float, float);
	void MSFiltering(Mat&);		
};
