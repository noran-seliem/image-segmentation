#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>



struct BGR_point{
  int b;
  int g;
  int r;
  int & operator[](int idx)
  {
    switch (idx)
    {
      case 0:
      return b;
      case 1:
      return g;
      case 2:
      return r;
    }
  }
};


typedef struct{
    std::vector<BGR_point> centroids;
    std::vector<int> clusters;
    cv::Mat segmented_image;
} KMeans_result;

std::vector<BGR_point> createData(cv::Mat image);
std::vector<BGR_point> RandomCentroidsGenerator(int number_of_centroids, cv::Mat image);
void getCentroids(std::vector<BGR_point> centroids);
double BGR_distance(BGR_point point1, BGR_point point2);
int getClosestCluster(BGR_point point, std::vector<BGR_point> centroids);
std::vector<int> assignClusters(std::vector<BGR_point> image_data, std::vector<BGR_point> centroids);
std::vector<BGR_point> updateCentroids(std::vector<BGR_point> image_data, std::vector<int> clusters, std::vector<BGR_point> centroids);
bool convState(std::vector<BGR_point> old_centroids, std::vector<BGR_point> new_centroids);
void segmentImage(cv::Mat image, KMeans_result &kmeans);
KMeans_result applyKmeans(cv::Mat image, int num_clusters, int max_iterations);

#endif //KMEANS_H