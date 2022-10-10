#include "Meanshift.hpp"

PixelParameters::PixelParameters()
{
	/* Initialize (x, y) of image by the end of the image */
	x = -1;
	y = -1;
}

void PixelParameters::PointLab()
{
	/* LAB Color Space Equation */
	l = l * 100 / 255;
	a = a - 128;
	b = b - 128;
}

void PixelParameters::PointRGB()
{
	/* RGB Color Space Equation */
	l = l * 255 / 100;
	a = a + 128;
	b = b + 128;
}

void PixelParameters::AccumulatePoints(PixelParameters Pt)
{
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	a += Pt.a;
	b += Pt.b;
}

void PixelParameters::CopyPoint(PixelParameters Pt)
{
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	a = Pt.a;
	b = Pt.b;
}

float PixelParameters::ColorDistance(PixelParameters Pt)
{
	return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

float PixelParameters::SpatialDistance(PixelParameters Pt)
{
	return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

void PixelParameters::PointScale(float scale)
{
	x *= scale;
	y *= scale;
	l *= scale;
	a *= scale;
	b *= scale;
}

void PixelParameters::SetPointValue(float px, float py, float pl, float pa, float pb)
{
	x = px;
	y = py;
	l = pl;
	a = pa;
	b = pb;
}

MeanShift::MeanShift(float s, float r) {
	SpatialRadius = s;
	ColorRadius = r;
}

void MeanShift::MSFiltering(Mat& Img) {
	int ROWS = Img.rows;			
	int COLS = Img.cols;			
	split(Img, IMGChannels);		// Split Image colors

	PixelParameters PtCur;					// Current point
	PixelParameters PtPrev;					// Previous point
	PixelParameters PtSum;					// Sum vector of the shift vector
	PixelParameters Pt;
	int Left;						// Left boundary
	int Right;						// Right boundary
	int Top;						// Top boundary
	int Bottom;						// Bottom boundary
	int NumPts;						// number of points in a hypersphere
	int step;

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) 
		{
			// *** Get the boundaries of the filter ***
			Left = (j - SpatialRadius) > 0 ? (j - SpatialRadius) : 0;		
			Right = (j + SpatialRadius) < COLS ? (j + SpatialRadius) : COLS;				
			Top = (i - SpatialRadius) > 0 ? (i - SpatialRadius) : 0;						
			Bottom = (i + SpatialRadius) < ROWS ? (i + SpatialRadius) : ROWS;	

			// Set current point and scale it to Lab color range
			PtCur.SetPointValue(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();
			step = 0;				// Step Counter

			do {
				PtPrev.CopyPoint(PtCur);						// Set the original point and previous one
				PtSum.SetPointValue(0, 0, 0, 0, 0);					// Initial Sum vector
				NumPts = 0;											// Count number of points that satisfy the bandwidths
				for (int hx = Top; hx < Bottom; hx++) {
					for (int hy = Left; hy < Right; hy++) {
						// Set point in the spatial bandwidth
						Pt.SetPointValue(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();

						// Check it satisfied color bandwidth or not
						if (Pt.ColorDistance(PtCur) < ColorRadius) {
							PtSum.AccumulatePoints(Pt);				// Accumulate the point to Sum vector
							NumPts++;								// Count
						}
					}
				}
				PtSum.PointScale(1.0 / NumPts);					// Scale Sum vector to average vector
				PtCur.CopyPoint(PtSum);							// Get new origin point
				step++;												

			// filter iteration to end
			} while ((PtCur.ColorDistance(PtPrev) > 0.3) && (PtCur.SpatialDistance(PtPrev) > 0.3) && (step < 5));

			// Scale the color
			PtCur.PointRGB();
			// Copy the result to image
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}
}