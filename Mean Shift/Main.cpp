#include"Meanshift.hpp"

int main()
{
	MeanShift M(.005, 45);
	Mat Image = imread("C:\\Users\\Mahdy\\Pictures\\Homes2.jpg");
	imshow("Original Image", Image);
	M.MSFiltering(Image);
	imshow("Result Image", Image);
	waitKey(0);
}
