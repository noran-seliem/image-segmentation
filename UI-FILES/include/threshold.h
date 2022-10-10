#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<numeric>
#include <cmath>

using namespace std;
typedef enum{BINARIZE,MASK}Mode;
typedef enum{OTSU,OPTIMAL}TMode;
#define PI (3.14)
void Binarize(cv::Mat&,cv::Mat&,int T);
void Mask(cv::Mat&,cv::Mat&,int T);
void calcHist(cv::Mat&,std::vector<int>&);
void classVar(std::vector<int>&,std::vector<float>&);
int  calcThreshold(std::vector<float>&);

void otsu(cv::Mat&,cv::Mat&,Mode);
void optimalThreshold(cv::Mat&,cv::Mat&,Mode);
void localThreshold(cv::Mat&,cv::Mat&,int,TMode,Mode);
