#include <iostream>
#include <QtWidgets\qapplication.h>
#include <qgraphicsview.h>
#include <qgraphicsitem.h>
#include <qgraphicseffect.h>
#include <qmessagebox.h>
#include <qobject.h>
#include <QApplication>
#include <QGraphicsEllipseItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <qdatetime.h>
#include <qmainwindow.h>
#include <qstatusbar.h>
#include <qmessagebox.h>
#include <qmenubar.h>
#include <qapplication.h>
#include <qpainter.h>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <qlabel.h>
#include <qimage.h>
#include <qpixmap.h>
#include <QMouseEvent>
#include <QStyleOptionGraphicsItem>
#include <qdebug.h>
#include <stdlib.h>
#include <mainwindow.h>
#include <qmainwindow.h>
#include <opencv2\ximgproc.hpp>


using namespace cv;
using namespace std;

struct CornerParameters {
	int x;
	int y;
	double r;
};

struct Harris {
	Mat R;
	std::vector<CornerParameters> points;
};
struct Keypoint {
	// discrete coordinates
	int i;
	int j;
	int octave;
	int scale; //index of gaussian image inside the octave

	int DOGIndex;
	// continuous coordinates (interpolated)
	float x;
	float y;
	float sigma;
	int ZOrientation;
	float extremum_val; //value of interpolated DoG extremum
	int RadiusOfCyrle;
	std::array<uint8_t, 128> descriptor;

	std::vector<std::pair<int, int>>Orientation;
};


struct Pyramids
{
	int Octave_Numbers;
	int ImageForOctave;
	std::vector<std::vector<cv::Mat>>Octaves;

};









/*
#define M_PI 3.1453674253




const int MAX_REFINEMENT_ITERS = 5;
const float SIGMA_MIN = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = 8;
const int N_SPO = 3;
const float C_DOG = 0.015;
const float C_EDGE = 10;

// computation of the SIFT descriptor
const int N_BINS = 36;
const float LAMBDA_ORI = 1.5;
const int N_HIST = 4;
const int N_ORI = 8;
const float LAMBDA_DESC = 6;

// feature matching
const float THRESH_ABSOLUTE = 350;
const float THRESH_RELATIVE = 0.7;





// First we made the struct pyramid that will contain all diffrent octaves and its number as well
// which will save alot of coding in process of back function to function
struct Pyramids
{
	int Octave_Numbers;
	int ImageForOctave;
	std::vector<std::vector<cv::Mat>>Octaves;

};


int KernalSize = 3;

struct Keypoint {
	// discrete coordinates
	int i;
	int j;
	int octave;
	int scale; //index of gaussian image inside the octave

	// continuous coordinates (interpolated)
	float x;
	float y;
	float sigma;
	float extremum_val; //value of interpolated DoG extremum

	std::array<uint8_t, 128> descriptor;
};

// The purpose of this function is to get the Octaves of the series of gaussin and then get DOG
// in Next function and this will be done by first 
Pyramids generate_gaussian_pyramid(cv::Mat Image, float SigmaMin, int NumOctaves, int ScalePerOctave)
{
	// we want to get what is the sigma as there will be vector of them for each gaussin to achieve the 
	// full octave

	float base_sigma = SigmaMin / MIN_PIX_DIST;
	cv::Mat base_img; 
	cv::resize(Image, base_img, cv::Size(Image.cols * 2, Image.rows * 2), cv::INTER_LINEAR);
	float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
	cv::GaussianBlur(base_img, base_img,cv::Size( KernalSize, KernalSize),sigma_diff);
	
	int imgs_per_octave = ScalePerOctave + 3;
	// determine sigma values for bluring
	float k = std::pow(2, 1.0 / ScalePerOctave);
	std::vector<float> sigma_vals{ base_sigma };
	for (int i = 1; i < imgs_per_octave; i++) {
		float sigma_prev = base_sigma * std::pow(k, i - 1);
		float sigma_total = k * sigma_prev;
		sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev * sigma_prev));
	}

	// create a scale space pyramid of gaussian images
	// images in each octave are half the size of images in the previous one
	Pyramids pyramid = {
		NumOctaves,
		imgs_per_octave,
		std::vector<std::vector<cv::Mat>>(NumOctaves)
	};

	for (int i = 0; i < NumOctaves; i++) {
		pyramid.Octaves[i].reserve(imgs_per_octave);
		pyramid.Octaves[i].push_back(std::move(base_img));
		for (int j = 1; j < sigma_vals.size(); j++) {
			const cv::Mat prev_img = pyramid.Octaves[i].back();

			
			cv::GaussianBlur(prev_img, prev_img, cv::Size(KernalSize, KernalSize), sigma_vals[j]);
			pyramid.Octaves[i].push_back(prev_img);
		}
		// prepare base image for next octave
		const cv::Mat next_base_img = pyramid.Octaves[i][imgs_per_octave - 3];

		cv::resize(next_base_img, next_base_img, cv::Size(next_base_img.cols / 2, next_base_img.rows/ 2));
	}
	return pyramid;

}


Pyramids generate_dog_pyramid(const Pyramids& img_pyramid)
{
		Pyramids dog_pyramid = {
		img_pyramid.Octave_Numbers,
		img_pyramid.ImageForOctave - 1,
		std::vector<std::vector<cv::Mat>>(img_pyramid.Octave_Numbers)
	};
	for (int i = 0; i < dog_pyramid.Octave_Numbers; i++) {
		dog_pyramid.Octaves[i].reserve(dog_pyramid.ImageForOctave);
		for (int j = 1; j < img_pyramid.ImageForOctave; j++) {
			cv::Mat diff = img_pyramid.Octaves[i][j];
			for (int pix_idx = 0; pix_idx < diff.rows*diff.cols; pix_idx++) {
				diff.data[pix_idx] -= img_pyramid.Octaves[i][j - 1].data[pix_idx];
			}
			dog_pyramid.Octaves[i].push_back(diff);
		}
	}
	return dog_pyramid;
}




bool point_is_extremum(const std::vector<cv::Mat> &octave, int scale, int x, int y)
{
	cv::Mat  img = octave[scale];
	cv::Mat  prev = octave[scale - 1];
	cv::Mat  next = octave[scale + 1];

	bool is_min = true, is_max = true;
	float val = img.at<float>(x, y);

	float neighbor;

	for (int dx : {-1, 0, 1}) {
		for (int dy : {-1, 0, 1}) {
			neighbor = prev.at<float>(x+dx, y+dy);

			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			
			neighbor = next.at<float>(x + dx, y + dy);

			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			neighbor = img.at<float>(x + dx, y + dy);

			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			if (!is_min && !is_max) return false;
		}
	}
	return true;
}



std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
	const std::vector<cv::Mat>& octave,
	int scale , float &offset_s, float & offset_x, float & offset_y)
{
	const cv::Mat img = octave[scale];
	const cv::Mat prev = octave[scale - 1];
	const cv::Mat next = octave[scale + 1];

	float g1, g2, g3;
	float h11, h12, h13, h22, h23, h33;
	int x = kp.i, y = kp.j;

	// gradient 
	g1 = (next.at<float>(x, y, 0) - prev.at<float>(x, y, 0)) * 0.5;
	g2 = (img.at<float>(x + 1, y, 0) - img.at<float>(x - 1, y, 0)) * 0.5;
	g3 = (img.at<float>(x, y + 1, 0) - img.at<float>(x, y - 1, 0)) * 0.5;

	// hessian
	h11 = next.at<float>(x, y, 0) + prev.at<float>(x, y, 0) - 2 * img.at<float>(x, y, 0);
	h22 = img.at<float>(x + 1, y, 0) + img.at<float>(x - 1, y, 0) - 2 * img.at<float>(x, y, 0);
	h33 = img.at<float>(x, y + 1, 0) + img.at<float>(x, y - 1, 0) - 2 * img.at<float>(x, y, 0);
	h12 = (next.at<float>(x + 1, y, 0) - next.at<float>(x - 1, y, 0)
		- prev.at<float>(x + 1, y, 0) + prev.at<float>(x - 1, y, 0)) * 0.25;
	h13 = (next.at<float>(x, y + 1, 0) - next.at<float>(x, y - 1, 0)
		- prev.at<float>(x, y + 1, 0) + prev.at<float>(x, y - 1, 0)) * 0.25;
	h23 = (img.at<float>(x + 1, y + 1, 0) - img.at<float>(x + 1, y - 1, 0)
		- img.at<float>(x - 1, y + 1, 0) + img.at<float>(x - 1, y - 1, 0)) * 0.25;

	// invert hessian
	float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
	float det = h11 * h22*h33 - h11 * h23*h23 - h12 * h12*h33 + 2 * h12*h13*h23 - h13 * h13*h22;
	hinv11 = (h22*h33 - h23 * h23) / det;
	hinv12 = (h13*h23 - h12 * h33) / det;
	hinv13 = (h12*h23 - h13 * h22) / det;
	hinv22 = (h11*h33 - h13 * h13) / det;
	hinv23 = (h12*h13 - h11 * h23) / det;
	hinv33 = (h11*h22 - h12 * h12) / det;

	// find offsets of the interpolated extremum from the discrete extremum
	offset_s = -hinv11 * g1 - hinv12 * g2 - hinv13 * g3;
	offset_x = -hinv12 * g1 - hinv22 * g2 - hinv23 * g3;
	offset_y = -hinv13 * g1 - hinv23 * g3 - hinv33 * g3;

	float interpolated_extrema_val = img.at<float>(x, y, 0)
		+ 0.5*(g1*offset_s + g2 * offset_x + g3 * offset_y);
	kp.extremum_val = interpolated_extrema_val;
	return { offset_s, offset_x, offset_y };
}





bool point_is_on_edge(const Keypoint& kp, const std::vector<cv::Mat>& octave, float edge_thresh = C_EDGE)
{
	cv::Mat  img = octave[kp.scale];
	float h11, h12, h22;
	int x = kp.i, y = kp.j;
	h11 = img.at<float>(x + 1, y, 0) + img.at<float>(x - 1, y, 0) - 2 * img.at<float>(x, y, 0);
	h22 = img.at<float>(x, y + 1, 0) + img.at<float>(x, y - 1, 0) - 2 * img.at<float>(x, y, 0);
	h12 = (img.at<float>(x + 1, y + 1, 0) - img.at<float>(x + 1, y - 1, 0)
		- img.at<float>(x - 1, y + 1, 0) + img.at<float>(x - 1, y - 1, 0)) * 0.25;

	float det_hessian = h11 * h22 - h12 * h12;
	float tr_hessian = h11 + h22;
	float edgeness = tr_hessian * tr_hessian / det_hessian;

	if (edgeness > std::pow(edge_thresh + 1, 2) / edge_thresh)
		return true;
	else
		return false;
}




void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
	float sigma_min = SIGMA_MIN,
	float min_pix_dist = MIN_PIX_DIST, int n_spo = N_SPO)
{
	kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s + kp.scale) / n_spo);
	kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x + kp.i);
	kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y + kp.j);
}


bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<cv::Mat>& octave,
	float contrast_thresh, float edge_thresh)
{
	int k = 0;
	bool kp_is_valid = false;
	while (k++ < MAX_REFINEMENT_ITERS) {
		float offset_s, offset_x, offset_y;
		fit_quadratic(kp, octave, kp.scale, offset_s, offset_x, offset_y);

		float max_offset = std::max({ std::abs(offset_s),
									 std::abs(offset_x),
									 std::abs(offset_y) });
		// find nearest discrete coordinates
		kp.scale += std::round(offset_s);
		kp.i += std::round(offset_x);
		kp.j += std::round(offset_y);
		if (kp.scale >= octave.size() - 1 || kp.scale < 1)
			break;

		bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
		if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
			find_input_img_coords(kp, offset_s, offset_x, offset_y);
			kp_is_valid = true;
			break;
		}
	}
	return kp_is_valid;
}



std::vector<Keypoint> find_keypoints(const Pyramids& dog_pyramid, float contrast_thresh,
	float edge_thresh)
{
	std::vector<Keypoint> keypoints;
	for (int i = 0; i < dog_pyramid.Octave_Numbers; i++) {
		const std::vector<cv::Mat>& octave = dog_pyramid.Octaves[i];
		for (int j = 1; j < dog_pyramid.ImageForOctave - 1; j++) {
			const cv::Mat  img = octave[j];
			for (int y = 1; y < img.rows - 1; y++) {
				for (int x = 1; x < img.cols - 1; x++) {
					if (std::abs(img.at<float>(x, y, 0)) < 0.8*contrast_thresh) {
						continue;
					}
					if (point_is_extremum(octave, j, x, y)) {
						Keypoint kp = { x, y, i, j, -1, -1, -1, -1 };
						bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
							edge_thresh);
						if (kp_is_valid) {
							keypoints.push_back(kp);
						}
					}
				}
			}
		}
	}
	return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
Pyramids generate_gradient_pyramid(const Pyramids& pyramid)
{
		Pyramids grad_pyramid = {
		pyramid.Octave_Numbers,
		pyramid.ImageForOctave,
		std::vector<std::vector<cv::Mat>>(pyramid.Octave_Numbers)
	};
	for (int i = 0; i < pyramid.Octave_Numbers; i++) {
		grad_pyramid.Octaves[i].reserve(grad_pyramid.ImageForOctave);
		int width = pyramid.Octaves[i][0].cols;
		int height = pyramid.Octaves[i][0].rows;
		for (int j = 0; j < pyramid.ImageForOctave; j++) {
			cv::Mat grad(width, height, 2);
			float gx, gy;
			for (int y = 1; y < grad.rows - 1; y++) {
				for (int x = 1; x < grad.rows - 1; x++) {
					gx = (pyramid.Octaves[i][j].at<float>(x + 1, y, 0)
						- pyramid.Octaves[i][j].at<float>(x - 1, y, 0)) * 0.5;
					grad.at<float>(x, y, 0)=gx;
					gy = (pyramid.Octaves[i][j].at<float>(x, y + 1, 0)
						- pyramid.Octaves[i][j].at<float>(x, y - 1, 0)) * 0.5;
					grad.at<float>(x, y, 1)=gy;
				}
			}
			grad_pyramid.Octaves[i].push_back(grad);
		}
	}
	return grad_pyramid;
}




// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
	float tmp_hist[N_BINS];
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < N_BINS; j++) {
			int prev_idx = (j - 1 + N_BINS) % N_BINS;
			int next_idx = (j + 1) % N_BINS;
			tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
		}
		for (int j = 0; j < N_BINS; j++) {
			hist[j] = tmp_hist[j];
		}
	}
}


std::vector<float> find_keypoint_orientations(Keypoint& kp,
	const Pyramids& grad_pyramid,
	float lambda_ori, float lambda_desc)
{
	float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
	const cv::Mat  img_grad = grad_pyramid.Octaves[kp.octave][kp.scale];

	// discard kp if too close to image borders 
	float min_dist_from_border = std::min({ kp.x, kp.y, pix_dist*img_grad.cols - kp.x,
										   pix_dist*img_grad.rows - kp.y });
	if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
		return {};
	}

	float hist[N_BINS] = {};
	int bin;
	float gx, gy, grad_norm, weight, theta;
	float patch_sigma = lambda_ori * kp.sigma;
	float patch_radius = 3 * patch_sigma;
	int x_start = std::round((kp.x - patch_radius) / pix_dist);
	int x_end = std::round((kp.x + patch_radius) / pix_dist);
	int y_start = std::round((kp.y - patch_radius) / pix_dist);
	int y_end = std::round((kp.y + patch_radius) / pix_dist);

	// accumulate gradients in orientation histogram
	for (int x = x_start; x <= x_end; x++) {
		for (int y = y_start; y <= y_end; y++) {
			gx = img_grad.at<float>(x, y, 0);
			gy = img_grad.at<float>(x, y, 1);
			grad_norm = std::sqrt(gx*gx + gy * gy);
			weight = std::exp(-(std::pow(x*pix_dist - kp.x, 2) + std::pow(y*pix_dist - kp.y, 2))
				/ (2 * patch_sigma*patch_sigma));
			theta = std::fmod(std::atan2(gy, gx) + 2 * M_PI, 2 * M_PI);
			bin = (int)std::round(N_BINS / (2 * M_PI)*theta) % N_BINS;
			hist[bin] += weight * grad_norm;
		}
	}

	smooth_histogram(hist);

	// extract reference orientations
	float ori_thresh = 0.8, ori_max = 0;
	std::vector<float> orientations;
	for (int j = 0; j < N_BINS; j++) {
		if (hist[j] > ori_max) {
			ori_max = hist[j];
		}
	}
	for (int j = 0; j < N_BINS; j++) {
		if (hist[j] >= ori_thresh * ori_max) {
			float prev = hist[(j - 1 + N_BINS) % N_BINS], next = hist[(j + 1) % N_BINS];
			if (prev > hist[j] || next > hist[j])
				continue;
			float theta = 2 * M_PI*(j + 1) / N_BINS + M_PI / N_BINS * (prev - next) / (prev - 2 * hist[j] + next);
			orientations.push_back(theta);
		}
	}
	return orientations;
}


void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
	float contrib, float theta_mn, float lambda_desc)
{
	float x_i, y_j;
	for (int i = 1; i <= N_HIST; i++) {
		x_i = (i - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
		if (std::abs(x_i - x) > 2 * lambda_desc / N_HIST)
			continue;
		for (int j = 1; j <= N_HIST; j++) {
			y_j = (j - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
			if (std::abs(y_j - y) > 2 * lambda_desc / N_HIST)
				continue;

			float hist_weight = (1 - N_HIST * 0.5 / lambda_desc * std::abs(x_i - x))
				*(1 - N_HIST * 0.5 / lambda_desc * std::abs(y_j - y));

			for (int k = 1; k <= N_ORI; k++) {
				float theta_k = 2 * M_PI*(k - 1) / N_ORI;
				float theta_diff = std::fmod(theta_k - theta_mn + 2 * M_PI, 2 * M_PI);
				if (std::abs(theta_diff) >= 2 * M_PI / N_ORI)
					continue;
				float bin_weight = 1 - N_ORI * 0.5 / M_PI * std::abs(theta_diff);
				hist[i - 1][j - 1][k - 1] += hist_weight * bin_weight*contrib;
			}
		}
	}
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
	int size = N_HIST * N_HIST*N_ORI;
	float *hist = reinterpret_cast<float *>(histograms);

	float norm = 0;
	for (int i = 0; i < size; i++) {
		norm += hist[i] * hist[i];
	}
	norm = std::sqrt(norm);
	float norm2 = 0;
	for (int i = 0; i < size; i++) {
		hist[i] = std::min(hist[i], 0.2f*norm);
		norm2 += hist[i] * hist[i];
	}
	norm2 = std::sqrt(norm2);
	for (int i = 0; i < size; i++) {
		float val = std::floor(512 * hist[i] / norm2);
		feature_vec[i] = std::min((int)val, 255);
	}
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
	const Pyramids& grad_pyramid,
	float lambda_desc)
{
	float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
	const cv::Mat  img_grad = grad_pyramid.Octaves[kp.octave][kp.scale];
	float histograms[N_HIST][N_HIST][N_ORI] = { 0 };

	//find start and end coords for loops over image patch
	float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST + 1.) / N_HIST;
	int x_start = std::round((kp.x - half_size) / pix_dist);
	int x_end = std::round((kp.x + half_size) / pix_dist);
	int y_start = std::round((kp.y - half_size) / pix_dist);
	int y_end = std::round((kp.y + half_size) / pix_dist);

	float cos_t = std::cos(theta), sin_t = std::sin(theta);
	float patch_sigma = lambda_desc * kp.sigma;
	//accumulate samples into histograms
	for (int m = x_start; m <= x_end; m++) {
		for (int n = y_start; n <= y_end; n++) {
			// find normalized coords w.r.t. kp position and reference orientation
			float x = ((m*pix_dist - kp.x)*cos_t
				+ (n*pix_dist - kp.y)*sin_t) / kp.sigma;
			float y = (-(m*pix_dist - kp.x)*sin_t
				+ (n*pix_dist - kp.y)*cos_t) / kp.sigma;

			// verify (x, y) is inside the description patch
			if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST + 1.) / N_HIST)
				continue;

			float gx = img_grad.at<float>(m, n, 0), gy = img_grad.at<float>(m, n, 1);
			float theta_mn = std::fmod(std::atan2(gy, gx) - theta + 4 * M_PI, 2 * M_PI);
			float grad_norm = std::sqrt(gx*gx + gy * gy);
			float weight = std::exp(-(std::pow(m*pix_dist - kp.x, 2) + std::pow(n*pix_dist - kp.y, 2))
				/ (2 * patch_sigma*patch_sigma));
			float contribution = weight * grad_norm;

			update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
		}
	}

	// build feature vector (descriptor) from histograms
	hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const cv::Mat& img, float sigma_min,
	int num_octaves, int scales_per_octave,
	float contrast_thresh, float edge_thresh,
	float lambda_ori, float lambda_desc)
{

	assert(img.channels() == 1 || img.channels() == 3);

	cv::Mat cloner = img.clone();
	
	cv::Mat img2;
	cv::cvtColor(cloner, img2, cv::COLOR_RGB2GRAY);


	const cv::Mat& input = img.channels() == 1 ? img : img2;
	Pyramids gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,scales_per_octave);
	Pyramids dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
	std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
	Pyramids grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);

	std::vector<Keypoint> kps;

	for (Keypoint& kp_tmp : tmp_kps) {
		std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
			lambda_ori, lambda_desc);
		for (float theta : orientations) {
			Keypoint kp = kp_tmp;
			compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
			kps.push_back(kp);
		}
	}
	return kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
	float dist = 0;
	for (int i = 0; i < 128; i++) {
		int di = (int)a[i] - b[i];
		dist += di * di;
	}
	return std::sqrt(dist);
}

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
	std::vector<Keypoint>& b,
	float thresh_relative,
	float thresh_absolute)
{
	assert(a.size() >= 2 && b.size() >= 2);

	std::vector<std::pair<int, int>> matches;

	for (int i = 0; i < a.size(); i++) {
		// find two nearest neighbours in b for current keypoint from a
		int nn1_idx = -1;
		float nn1_dist = 100000000, nn2_dist = 100000000;
		for (int j = 0; j < b.size(); j++) {
			float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
			if (dist < nn1_dist) {
				nn2_dist = nn1_dist;
				nn1_dist = dist;
				nn1_idx = j;
			}
			else if (nn1_dist <= dist && dist < nn2_dist) {
				nn2_dist = dist;
			}
		}
		if (nn1_dist < thresh_relative*nn2_dist && nn1_dist < thresh_absolute) {
			matches.push_back({ i, nn1_idx });
		}
	}
	return matches;
}

cv::Mat draw_keypoints(const cv::Mat& img, const std::vector<Keypoint>& kps)
{
	cv::Mat res(img);
	if (img.channels() == 1) {
		cv::Mat cloner = res.clone();
		cv::cvtColor(cloner, res, cv::COLOR_RGB2GRAY);
	}
	for (auto& kp : kps) {
		cv::circle(res,cv::Point( kp.x, kp.y), 5,cv::Scalar(255,0,0));
	}
	return res;
}

cv::Mat draw_matches(const cv::Mat& a, const cv::Mat& b, std::vector<Keypoint>& kps_a,
	std::vector<Keypoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
	cv::Mat res(a.cols + b.cols, std::max(a.rows, b.rows), 3);

	for (int i = 0; i < a.cols; i++) {
		for (int j = 0; j < a.rows; j++) {
			res.at<float>(i, j, 0)= a.at<float>(i, j, 0);
			res.at<float>(i, j, 1)= a.at<float>(i, j, a.channels() == 3 ? 1 : 0);
			res.at<float>(i, j, 2)= a.at<float>(i, j, a.channels() == 3 ? 2 : 0);
		}
	}
	for (int i = 0; i < b.cols; i++) {
		for (int j = 0; j < b.rows; j++) {
			res.at<float>(a.cols + i, j, 0)= b.at<float>(i, j, 0);
			res.at<float>(a.cols + i, j, 1)= b.at<float>(i, j, b.channels() == 3 ? 1 : 0);
			res.at<float>(a.cols + i, j, 2)= b.at<float>(i, j, b.channels() == 3 ? 2 : 0);
		}
	}

	for (auto& m : matches) {
		Keypoint& kp_a = kps_a[m.first];
		Keypoint& kp_b = kps_b[m.second];
		cv::line(res,cv::Point(kp_a.x, kp_a.y),cv::Point(a.cols + kp_b.x, kp_b.y),cv::Scalar(255,0,0));
	}
	return res;
}




*/


float Orientation(int x,int y) 
{


	return 0.2;
	
}



// The purpose of this function is to make a very big continued histogram disspiated in 4 parts'
// these 4 parts each will have thier unique histogram 
// after that they will be attached to each other and be the deminer of 
// histogram orientation
std::vector<std::vector<int>> dessiptor(std::vector<Keypoint>Keypointss)
{

	// so we make first 

	// 1/4 of histogram which will be cleared after each quarter to add the new one
	std::vector<int>Histogram;

	// the Big Histo of 4 added together
	std::vector<int>HISTO;
	//the vector that will contation this big histo for each Keypoint
	std::vector<std::vector<int>>KeyPointHisto;
	std::vector<std::vector<int>>VecHistogram;

	// first inichialize
	for (int i = 0; i < 360;i++) 
	{
		Histogram.push_back(0);
	
	}

	for (int i = 0; i < Keypointss.size();i++) 
	{

		// the four quarter starter and ender ... which should be deternined by radius of the crrcle given before
		int xStart = std::floor(Keypointss[i].x-Keypointss[i].RadiusOfCyrle);
		int xEnd = std::floor(Keypointss[i].x + Keypointss[i].RadiusOfCyrle);
		int YStart = std::floor(Keypointss[i].y - Keypointss[i].RadiusOfCyrle);
		int YEnd = std::floor(Keypointss[i].y + Keypointss[i].RadiusOfCyrle);

		


		for (int y = YStart; y < YEnd/2;y++) 
		{
			for (int x = xStart;x < xEnd/2;x++) 
			{
		
				// again this is histogram for orientation of specific point which we will get from function
				Histogram[(int)Orientation(x, y)]++;

			
			}
		}

		VecHistogram.push_back(Histogram);
		
		for (int i = 0; i < Histogram.size();i++) 
		{
			HISTO.push_back(Histogram[i]);
		
		}

		Histogram.clear();



		for (int y = YEnd/2; y < YEnd;y++)
		{
			for (int x = xStart;x < xEnd/2;x++)
			{

				Histogram[(int)Orientation(x, y)]++;



			}
		}


		VecHistogram.push_back(Histogram);
		for (int i = 0; i < Histogram.size();i++)
		{
			HISTO.push_back(Histogram[i]);

		}

		
		Histogram.clear();

		for (int y = YStart; y < YEnd/2;y++)
		{
			for (int x = xEnd/2;x < xEnd;x++)
			{

				Histogram[(int)Orientation(x, y)]++;



			}
		}

		VecHistogram.push_back(Histogram);
		for (int i = 0; i < Histogram.size();i++)
		{
			HISTO.push_back(Histogram[i]);

		}

		Histogram.clear();

		for (int y = YEnd/2; y < YEnd;y++)
		{
			for (int x = xEnd / 2;x < xEnd ;x++)
			{

				Histogram[(int)Orientation(x, y)]++;



			}
		}



		VecHistogram.push_back(Histogram);
		for (int i = 0; i < Histogram.size();i++)
		{
			HISTO.push_back(Histogram[i]);

		}

		Histogram.clear();

		KeyPointHisto.push_back(HISTO);
	}


	// after the four quarters we will return the big histo
	return KeyPointHisto;
}



void SetOrientation(float angle, int x , int y) 
{

	// the set orientation should be given past x,t and thier orientation and new x,y according to the new orientation
	// retuen x , y

}


std::vector<Keypoint> Matching(std::vector<Keypoint>M1, std::vector<Keypoint>M2)
{

	// we  need to match the two points together and this is done by
	// first getting all the points histo vector as said before in both cases
	std::vector < std::vector<int >> HistoKeypointsOfM1=dessiptor(M1);
	std::vector < std::vector<int >> HistoKeypointsOfM2=dessiptor(M2);
	std::vector<Keypoint> LastResult;


	for (int i = 0; i < M1.size();i++) 
	{

		for (int z = 0; z < HistoKeypointsOfM2.size();z++) 
		{
			// then we see each point and its orientation is histogram and get the ZOrientation
			// which is the overall orientation of the image
			// after we - substract that overall orientation we will get the new destined 
			HistoKeypointsOfM2[z][i] = HistoKeypointsOfM2[z][i] - M1[i].ZOrientation;

			// which then should return new x,y according to the oriantaion
			SetOrientation(HistoKeypointsOfM2[z][i],M2[z].x,M2[z].y);


			for (int k = 0; k < HistoKeypointsOfM1.size();k++) 
			
			{
				//then compare after the orientation with original keypoint location if it is the same then 
				//push it if not ignore
				if (M2[z].x == M1[k].x && M2[z].y == M1[k].y) 
				{
				
					LastResult.push_back(M1[k]);
				}
			}
		}
	
	}
	

	//and return the matched keypoints to be placed in the end
	return LastResult;




}





Mat frame;
Point pt(-1, -1);
bool newCoords = false;

void mouse_callback(int  event, int  x, int  y, int  flag, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		// Store point coordinates
		pt.x = x;
		pt.y = y;
		newCoords = true;
	}
}

#include <iostream>
#include <cstdlib>


#define ever ;;

void RegioinGrowing(cv::Mat image,int threeshold, int XPoint , int YPoint) 
{





	// The way this function works is by the Getting the RGP of each point and make an estimate of the box neigbour values 
	// if the neighbours are near the value of the point (x,y) then add these points to the cluster
	// after adding these points go to the bigger box 
	// the way to get to bigger box is little tricky

	bool Pass = true;

	std::vector<int> XStorer;
	std::vector<int> YStorer;
	XStorer.push_back(XPoint);
	YStorer.push_back(YPoint);

	int zzzp = 0;
	int zk = 0;

	// we made a vectoer this vector is  is what makes the points 
	// meaning the cluster should stop in points that doesn't justify the threeshold
	// meaning this vector is the vectors of points that justify the threeshold from thier paast values
	// for example 
	// we first add our starter point 

	while (zk < XStorer.size()) {

		// then get a box of the values near

		for (int m = -1; m < 2;m++) {

			for (int n = -1; n < 2; n++) {

				// will explain this part in the end

				for (int e = 0;e < XStorer.size();e++)
				{
					if (XStorer[zk] + m == XStorer[e] && YStorer[zk] + n == YStorer[e]) { Pass = false; }

				}

				// get the color of both the value of the point and the neighbours
				Vec3b & color = image.at<Vec3b>(XStorer[zk] + m, YStorer[zk] + n);
				Vec3b & color2 = image.at<Vec3b>(XStorer[zk], YStorer[zk]);




				//then get the diffrence of the values
				int alpha = color[0] - color2[0];
				int Beta = color[1] - color2[1];
				int Gama = color[2] - color2[2];

				// make sure it is modulus number
				if (alpha <= 0)
				{
					alpha = alpha * (-1);

				}
				if (Beta <= 0)
				{
					Beta = Beta * (-1);

				}
				if (Gama <= 0)
				{
					Gama = Gama * (-1);

				}



				//then make threesholding if

				if ((alpha <= threeshold || Beta <= threeshold || Gama <= threeshold) && Pass)
				{
					// oh my
					// now we add to the XStorer 
					// meaning it will enter the loop again with z++ for the next point"SSSS"
					XStorer.push_back(XStorer[zk] + m);
					YStorer.push_back(YStorer[zk] + n);

					//then make these points appear
					Vec3b & Thecolor = image.at<Vec3b>(XStorer[zk] + m, YStorer[zk] + n);
					Thecolor[0] = 255;
					Thecolor[1] = 0;
					Thecolor[2] = 0;


					// I made the function overriding meaning when appling the function you can hit space to see how it works or set a limit

					image.at<Vec3b>(XStorer[zk] + m, YStorer[zk] + n) = Thecolor;
					
					cv::imshow("Image", image);
					cv::waitKey(2);
						
					
					//cv::waitKey();

				}

				// but wait 
				// IImaging 3*3 matrix where  the points was valid like this
				/*
				// 0   1   1 
				   0   0   1
				   0   0   0

				   Now we add the points of 1 to the vector but wait this 1 in the up right 
				   had its kernal like this
				   1   0			 1
				   0   0			 0
				   1   1(Repeated)   1

				   You see this one  is Repeated meaning it already clustered before so we need to PASS it. 
				   // meaning will make if Condition if we ever passed a point doon't go back agin to it.

				
				*/
				Pass = true;
			}
		}
		zk++;
	}






	cv::imshow("Image", image);
	cv::waitKey();




}


void Agglomerative(cv::Mat image2) 
{



	//Okay this function is super heavy in computinal wise
	// the way this cluster works is like the same but instaead 3 major things changes
	

	// first each pixel is clustered within pos , Values of RGB 
	// second with each pixel not just neighbours
	// the cluster is based as heical meaning after the first cluster 
	// 2 things happen 
	// One of the pixel get the cluster value
	// while the second disappear or not mentioned any more
	// so in running this code 
	/*
	The computinal is really heavy so need to not get the output but in steps
	
	*/
	cv::Mat imagePutter = image2.clone();
	imagePutter.setTo(cv::Scalar(0, 0, 0));
	std::vector< std::pair<int, int>> POS;
	std::vector<int>RVAlue;
	std::vector<int>BVAlue;
	std::vector<int>GVAlue;
	std::vector<int> TheMV;
	std::vector<std::pair<int, int>> TheTwoOnes;
	std::vector<int>ThePassVector;
	int TheIndexWanted;


	// each eimage values has 2 For loops one in pixel other comparing it with every other pixel EXCEPT ITSELF or 
	// it will always be min
	for (int y = 0; y < image2.cols;y++)
	{
		for (int x = 0; x < image2.rows;x++)
		{

			Vec3b & Mistercolor = image2.at<Vec3b>(x, y);
			POS.push_back(std::make_pair(x, y));
			RVAlue.push_back(Mistercolor[0]);
			GVAlue.push_back(Mistercolor[1]);
			BVAlue.push_back(Mistercolor[2]);







		}
	}

	int m = 0;
	//pow(pow(POS[z].first - POS[z].second, 2) +
	for (int z = 0;z < POS.size() - 1;z++)
	{
		TheMV.clear();
		TheTwoOnes.clear();


		// a break Point for limitations of clustering
		if (m >= 2000)
		{
			cv::imshow("ImagePuttr2", imagePutter);
			cv::waitKey();
			break;
		}

		for (int Index = 0; Index < POS.size() - 1;Index++)
		{


			for (int ZEER = 0; ZEER < ThePassVector.size();ZEER++)
			{
				if (Index == ThePassVector[ZEER])
				{
					Index++;

				}

				if (z == ThePassVector[ZEER])
				{
					z++;

				}



			}


			// now we want the min , and not just that we need the exact location of the two minimums

			if (Index != z)
			{

				float min_distance = pow(pow(RVAlue[z] - RVAlue[Index], 2) + pow(BVAlue[z] - BVAlue[Index], 2) + pow(GVAlue[z] - GVAlue[Index], 2) + pow(POS[z].first - POS[Index].first, 2) + pow(POS[z].second - POS[Index].second, 2), 0.5);


				TheMV.push_back(min_distance);

				TheTwoOnes.push_back(std::make_pair(z, Index));

			}
			else
			{
				continue;
			}


		}


		float TheminimumRGB = *min_element(TheMV.begin(), TheMV.end());
		for (int i = 0; i < TheMV.size() - 1; i++)
		{
			if (TheMV[i] == TheminimumRGB)
			{
				TheIndexWanted = i;

			}


		}

		QMessageBox mc;
		//mc.exec();    

		//POS.erase(POS.begin+ TheTwoOnes[TheIndexWanted].second-1);
		//RVAlue.erase(RVAlue.begin + TheTwoOnes[TheIndexWanted].second - 1);
		//GVAlue.erase(RVAlue.begin + TheTwoOnes[TheIndexWanted].second - 1);
		//BVAlue.erase(RVAlue.begin + TheTwoOnes[TheIndexWanted].second - 1);
		
		// let the min 2 Points to be the value of them and add them to new image which makes sence because the cluster that is done
		// will be used again to cluster the other while the other value stay the same 
		// meaning we have 4 points for example nearest + min tyo each other
		// the values will first cluster 1 , 2 making them the same color
		// then 2 gets ignoreed as oif it doersn't exist anymore because the two became one cluster
		// then 1 cluster with three making three has color differ from 1 so that means in this clusteringh
		// we will see not just the high end ' or low roots but  allllllllllllllllll wiull be seen.

		RVAlue[TheTwoOnes[TheIndexWanted].second] = (int)(RVAlue[TheTwoOnes[TheIndexWanted].second] + RVAlue[TheTwoOnes[TheIndexWanted].first]) / 2;
		GVAlue[TheTwoOnes[TheIndexWanted].second] = (int)(GVAlue[TheTwoOnes[TheIndexWanted].second] + GVAlue[TheTwoOnes[TheIndexWanted].first]) / 2;
		BVAlue[TheTwoOnes[TheIndexWanted].second] = (int)(BVAlue[TheTwoOnes[TheIndexWanted].second] + BVAlue[TheTwoOnes[TheIndexWanted].first]) / 2;

		Vec3b & Mistercolor2 = imagePutter.at<Vec3b>(POS[TheTwoOnes[TheIndexWanted].first].first, POS[TheTwoOnes[TheIndexWanted].first].second);
		Mistercolor2[0] = RVAlue[TheTwoOnes[TheIndexWanted].second];
		Mistercolor2[1] = GVAlue[TheTwoOnes[TheIndexWanted].second];
		Mistercolor2[2] = BVAlue[TheTwoOnes[TheIndexWanted].second];


		imagePutter.at<Vec3f>(POS[TheTwoOnes[TheIndexWanted].first].first, POS[TheTwoOnes[TheIndexWanted].first].second) = Mistercolor2;
		imagePutter.at<Vec3f>(POS[TheTwoOnes[TheIndexWanted].second].first, POS[TheTwoOnes[TheIndexWanted].second].second) = Mistercolor2;


		ThePassVector.push_back(TheTwoOnes[TheIndexWanted].second);
		cv::imshow("ImagePuttr23", imagePutter);
		cv::waitKey();

		m++;


	}

	




}







cv::Mat RGBToLUV(cv::Mat image) 
{



	cv::Mat dst=image.clone();
	
	cv::Mat Bands[3];
	Bands[0] = image.clone();
	Bands[1] = image.clone();
	Bands[2] = image.clone();
	(Bands+1)->setTo(cv::Scalar(0, 0, 0));
	Bands->setTo(cv::Scalar(0, 0, 0));
	(Bands+2)->setTo(cv::Scalar(0, 0, 0));

	cv::cvtColor(Bands[0], Bands[0], COLOR_RGB2GRAY);
	cv::cvtColor(Bands[1], Bands[1], COLOR_RGB2GRAY);
	cv::cvtColor(Bands[2], Bands[2], COLOR_RGB2GRAY);


	for (int i = 0; i < image.cols; i++)
		for (int j = 0; j < image.rows; j++)
		{


			Vec3b v3 = image.at<Vec3b>(i, j);
			float b = ((float)v3[0]) / 255;
			float g = ((float)v3[1]) / 255;
			float r = ((float)v3[2]) / 255;

			float x = r * 0.412453 + g * 0.357580 + b * 0.180423;
			float y = r * 0.212671 + g * 0.715160 + b * 0.072169;
			float z = r * 0.019334 + g * 0.119193 + b * 0.950227;

			//L
			if (y > 0.008856) {
				Bands[0].at<uchar>(i, j) = 255.0 / 100.0 * (116.0 * pow(y, 1.0 / 3.0));
				dst.at<Vec3b>(i, j)[0] = 255.0 / 100.0 * (116.0 * pow(y, 1.0 / 3.0));
				
			}
			else {
				Bands[0].at<uchar>(i, j) = 255.0 / 100.0 * (903.3 * y);
				dst.at<Vec3b>(i, j)[0] = 255.0 / 100.0 * (903.3 * y);
			}

			float u = 4 * x / (x + 15.0 * y + 3.0 * z);
			float v = 9 * y / (x + 15.0* y + 3.0 * z);
			
			//U
			Bands[1].at<uchar>(i, j) = 255.0 / 354.0 * (13 * Bands[0].at<uchar>(i, j)*(u - 0.19793943) + 134.0);
			
			dst.at<Vec3b>(i, j) = 255.0 / 354.0 * (13 * Bands[0].at<uchar>(i, j)*(u - 0.19793943) + 134.0);

			//v
			Bands[2].at<uchar>(i, j) = 255.0 / 262.0 * (13.0 * Bands[0].at<uchar>(i, j)*(v - 0.46831096) + 140.0);
			dst.at<Vec3b>(i, j) = 255.0 / 262.0 * (13.0 * Bands[0].at<uchar>(i, j)*(v - 0.46831096) + 140.0);
		}

	//vector<Mat> channels = { Bands[0],Bands[1],Bands[2] };
	//cv::merge(channels, dst);

	return dst;
}



int main(int argc, char * argv[])
{


	QApplication app(argc,argv);
	
	MainWindow t;
	t.show();
	app.exec();
}