#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <qfiledialog.h>

#include <qmessagebox.h>

#include <qgraphicsscene.h>
#include <qgraphicsview.h>

using namespace cv;
using namespace std;

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
#include <Meanshift.hpp>
#include <cvQTconvert.h>

#include <threshold.h>
#include <kmeans.h>
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	

	connect(ui->actionOpen, SIGNAL(triggered()), this, SLOT(OpenImage()));
	connect(ui->actionRegion_Groeing, SIGNAL(triggered()), this, SLOT(RegionGrowingCall()));
	connect(ui->actionAgglomerative, SIGNAL(triggered()), this, SLOT(AgglomerativeCall()));
	connect(ui->actionK_means, SIGNAL(triggered()), this, SLOT(K_MeansCall()));
	connect(ui->actionMean_Shift, SIGNAL(triggered()), this, SLOT(Mean_ShiftCall()));
	connect(ui->Global_Otsu, SIGNAL(triggered()), this, SLOT(_Otsu()));
	connect(ui->globalOptimal, SIGNAL(triggered()), this, SLOT(_OptimalThreshold()));
	connect(ui->LocalOptimal, SIGNAL(triggered()), this, SLOT(_LocalThresholdOptimal()));
	connect(ui->Local_Otsu, SIGNAL(triggered()), this, SLOT(_LocalThresholdOtsu()));

	
	
	


}


void MainWindow::_LocalThresholdOptimal()
{


	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);

	
	cv::Mat cloner;
	cloner = TheAppImage.clone();
	cvtColor(TheAppImage, TheAppImage, COLOR_RGB2GRAY);


	QMessageBox TimerMessage;
	auto t1 = high_resolution_clock::now();
	localThreshold(TheAppImage, TheAppImage, TheAppImage.rows*0.2, OPTIMAL, BINARIZE);

	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();



	QPixmap im = ASM::cvMatToQPixmap(TheAppImage);
	TheAppImage = cloner.clone();
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);


}
void MainWindow::_LocalThresholdOtsu()
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);
	cv::Mat cloner;
	cloner = TheAppImage.clone();

	cvtColor(TheAppImage, TheAppImage, COLOR_RGB2GRAY);
	
	
	auto t1 = high_resolution_clock::now();
	localThreshold(TheAppImage, TheAppImage, TheAppImage.rows*0.2, OTSU, BINARIZE);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();



	QPixmap im = ASM::cvMatToQPixmap(TheAppImage);
	TheAppImage = cloner.clone();
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);

}




void MainWindow::_OptimalThreshold() 
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);
	cv::Mat cloner;
	cloner = TheAppImage.clone();

	cvtColor(TheAppImage, TheAppImage, COLOR_RGB2GRAY);

	
	

	auto t1 = high_resolution_clock::now();
	optimalThreshold(TheAppImage, TheAppImage, BINARIZE);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();



	QPixmap im = ASM::cvMatToQPixmap(TheAppImage);
	TheAppImage = cloner.clone();
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
}


void MainWindow::_Otsu()
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);
	cv::Mat cloner;
	cloner = TheAppImage.clone();

	cvtColor(TheAppImage, TheAppImage, COLOR_RGB2GRAY);

	



	auto t1 = high_resolution_clock::now();
	otsu(TheAppImage, TheAppImage, BINARIZE);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();


	QPixmap im = ASM::cvMatToQPixmap(TheAppImage);
	TheAppImage = cloner.clone();
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
}








void MainWindow::Mean_ShiftCall() 
{


	
	auto t1 = high_resolution_clock::now();
	Mean_Shift(TheAppImage, .005, 45);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();

}

void MainWindow::Mean_Shift(cv::Mat Image, float s, float r)
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);

	MeanShift M(s, r);
	M.MSFiltering(Image);
	QPixmap im = ASM::cvMatToQPixmap(Image);
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);

}

void MainWindow::K_MeansCall() 
{
	


	auto t1 = high_resolution_clock::now();
	K_Means(TheAppImage);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	duration<double, std::milli> ms_double = t2 - t1;
	QMessageBox mr;
	mr.setText(QString::number(ms_double.count()));
	mr.exec();

}


void MainWindow::K_Means(cv::Mat  Image) 
{
	
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);

	KMeans_result result;
	result = applyKmeans(Image, 5, 10);

	
	QPixmap im = ASM::cvMatToQPixmap(result.segmented_image);
	scene->clear();
	scene->addPixmap(im);

	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
}



void MainWindow::OpenImage() 
{
	auto fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image"), "C://", tr("Image Files (*.png *.jpg *.bmp *.jpeg)"));
	TheAppImage = cv::imread(fileName.toStdString());
	setView1(TheAppImage);
	



}


void MainWindow::RegionGrowingCall() 
{

	RegionGrowing(TheAppImage, 100, 40, 70);

}

void MainWindow::AgglomerativeCall() 
{

	Agglomerative(TheAppImage);


}
void MainWindow::Agglomerative(cv::Mat image2)
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);

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


		QPixmap im = ASM::cvMatToQPixmap(imagePutter);
		scene->clear();
		scene->addPixmap(im);

		ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);

		//cv::imshow("Image", image);
		cv::waitKey(2);

		m++;


	}





}

void MainWindow::RegionGrowing(cv::Mat image, int threeshold, int XPoint, int YPoint)
{
	QGraphicsScene* scene = new QGraphicsScene;
	ui->graphicsView_2->setScene(scene);

	

	
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
					QPixmap im = ASM::cvMatToQPixmap(image);
					scene->clear();
					scene->addPixmap(im);

					ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
					
					//cv::imshow("Image", image);
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


void MainWindow::setView1(cv::Mat image) {
	QPixmap im = ASM::cvMatToQPixmap(image);
	QGraphicsScene* scene = new QGraphicsScene;
	scene->addPixmap(im);
	ui->graphicsView->setScene(scene);
	ui->graphicsView->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
}
void MainWindow::setView2(cv::Mat image) {
	QPixmap im = ASM::cvMatToQPixmap(image);
	QGraphicsScene* scene = new QGraphicsScene;
	scene->addPixmap(im);
	ui->graphicsView_2->setScene(scene);
	ui->graphicsView_2->fitInView(scene->sceneRect(), Qt::IgnoreAspectRatio);
}
MainWindow::~MainWindow()
{
    delete ui;
}




