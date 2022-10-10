#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include "kmeans.h"
#define MIN_CENTROID_DISTANCE 100

using namespace cv;


int main() {
    Mat img = imread("../images/beach.jpg");
    if (img.empty()) {
        std::cout << "Image File "
            << "Not Found" << std::endl;

        // wait for any key press
        std::cin.get();
        return -1;
    }
    imshow("image", img);
    KMeans_result result;
    result = applyKmeans(img, 5, 10);
    imshow("kmeans", result.segmented_image);
    waitKey( );
    return 0;

}

void getCentroids(std::vector<BGR_point> centroids) {
    for (auto& centroid : centroids) {
        std::cout << "(" << centroid.b << "," << centroid.g << "," << centroid.r << "), ";
    }
}

double BGR_distance(BGR_point p1, BGR_point p2) {
    // Changing this remember to change #define MIN_CENTROID_DISTANCE 100 if needed 
    double b_distance = (p1.b - p2.b) * (p1.b - p2.b);
    double g_distance = (p1.g - p2.g) * (p1.g - p2.g);
    double r_distance = (p1.r - p2.r) * (p1.r - p2.r);
    double distance = b_distance + g_distance + r_distance;
    return distance;
}

std::vector<BGR_point> RandomCentroidsGenerator(int number_of_centroids, cv::Mat image) {
    std::random_device rd;
    std::default_random_engine random_generator(rd());
    std::uniform_int_distribution<int> rows_distribution(0, image.rows);
    std::uniform_int_distribution<int> columns_distribution(0, image.cols);
    std::vector<BGR_point> centroids;
    for (int centroid_index = 0; centroid_index < number_of_centroids; centroid_index++) {
        bool centroid_far;
        BGR_point centroid;
        do {
            centroid_far = true;
            // std::cout << "Generating New Random Centroid ... ";
            int row = rows_distribution(random_generator);
            int col = columns_distribution(random_generator);
            centroid.b = image.at<cv::Vec3b>(row, col)[0];
            centroid.g = image.at<cv::Vec3b>(row, col)[1];
            centroid.r = image.at<cv::Vec3b>(row, col)[2];
            // std::cout << "(" << centroid.b << "," << centroid.g << "," << centroid.r << ")" << std::endl;
            for (int i = 0; i < centroids.size(); ++i) {
                double distance = BGR_distance(centroid, centroids[i]);
                // std::cout << "Distance: " << distance << std::endl;
                if (distance <= MIN_CENTROID_DISTANCE) {
                    // std::cout << "Distance is less than 5" << std::endl;
                    centroid_far = false;
                    break;
                }
            }
        } while (!centroid_far);
        // std::cout << "Centroid Pushed" << std::endl;
        centroids.push_back(centroid);
    }
    return centroids;
}

std::vector<BGR_point> createData(cv::Mat image) {
    int rows = image.rows;
    int cols = image.cols;
    std::vector<BGR_point> image_data;
    BGR_point point;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            point.b = image.at<cv::Vec3b>(r, c)[0];
            point.g = image.at<cv::Vec3b>(r, c)[1];
            point.r = image.at<cv::Vec3b>(r, c)[2];
            image_data.push_back(point);
        }
    }
    return image_data;
}


int getClosestCluster(BGR_point point, std::vector<BGR_point> centroids) {
    double min_distance = BGR_distance(point, centroids[0]);
    int closest_cluster = 0;
    double distance;
    for (int cluster = 1; cluster < centroids.size(); cluster++) {
        distance = BGR_distance(point, centroids[cluster]);
        if (distance < min_distance) {
            min_distance = distance;
            closest_cluster = cluster;
        }
    }
    return closest_cluster;
}

std::vector<int>assignClusters(std::vector<BGR_point> image_data, std::vector<BGR_point> centroids) {
    int n_samples = image_data.size();
    int cluster_index = 0;
    std::vector<int> clusters_vector;
    for (int i = 0; i < n_samples; i++) {
        cluster_index = getClosestCluster(image_data[i], centroids);
        clusters_vector.push_back(cluster_index);
    }
    return clusters_vector;
}

std::vector<BGR_point> updateCentroids(std::vector<BGR_point> image_data, std::vector<int> clusters, std::vector<BGR_point> centroids) {
    std::vector<BGR_point> new_centroids;
    for (int c = 0; c < centroids.size(); c++) {
        BGR_point point;
        point.b = 0;
        point.g = 0;
        point.r = 0;
        int cluster_count = 0;
        for (int im = 0; im < image_data.size(); im++) {
            if (clusters[im] == c) {
                point.b += image_data[im].b;
                point.g += image_data[im].g;
                point.r += image_data[im].r;
                cluster_count++;
            }
        }
        if (cluster_count > 0) {
            point.b /= cluster_count;
            point.g /= cluster_count;
            point.r /= cluster_count;
            new_centroids.push_back(point);
        }
        else {
            new_centroids.push_back(centroids[c]);
        }
    }
    return new_centroids;
}

bool convState(std::vector<BGR_point> old_centroids, std::vector<BGR_point> new_centroids) {
    double sum = 0.0;
    for (int cent_idx = 0; cent_idx < old_centroids.size(); ++cent_idx) {
        sum += std::pow((old_centroids[cent_idx].b - new_centroids[cent_idx].b), 2) + std::pow((old_centroids[cent_idx].g - new_centroids[cent_idx].g), 2) + std::pow((old_centroids[cent_idx].r - new_centroids[cent_idx].r), 2);
    }
    return sum < 1;
}

void segmentImage(cv::Mat image, KMeans_result& kmeans) {
    cv::Mat segmented_image = image.clone();
    int cols = image.cols;
    int rows = image.rows;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            segmented_image.at<cv::Vec3b>(r, c)[0] = kmeans.centroids[kmeans.clusters[r * cols + c]].b;
            segmented_image.at<cv::Vec3b>(r, c)[1] = kmeans.centroids[kmeans.clusters[r * cols + c]].g;
            segmented_image.at<cv::Vec3b>(r, c)[2] = kmeans.centroids[kmeans.clusters[r * cols + c]].r;
        }
    }
    kmeans.segmented_image = segmented_image;
}

KMeans_result applyKmeans(cv::Mat imageBGR, int num_clusters, int max_iterations) {
    cv::Mat image;
    if (imageBGR.channels() != 3) {
        cv::cvtColor(imageBGR, image, cv::COLOR_GRAY2BGR);
    }
    else {
        image = imageBGR.clone();
    }
    std::vector<BGR_point> centroids = RandomCentroidsGenerator(num_clusters, image);
    std::vector<BGR_point> image_data = createData(image);
    std::vector<int> clusters;
    std::vector<BGR_point> old_centroids;
    for (int iter = 1; iter <= max_iterations; ++iter) {
        clusters = assignClusters(image_data, centroids);
        old_centroids = centroids;
        centroids = updateCentroids(image_data, clusters, centroids);
        if (convState(old_centroids, centroids)) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    KMeans_result kmeans_result;
    kmeans_result.clusters = clusters;
    kmeans_result.centroids = centroids;
    segmentImage(image, kmeans_result);
    return kmeans_result;
}
