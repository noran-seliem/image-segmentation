#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
	
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
	cv::Mat TheAppImage;
	

private:
    Ui::MainWindow *ui;
	void setView1(cv::Mat image);
	void setView2(cv::Mat image);
	
private slots:
	void OpenImage();
	void RegionGrowingCall();
	void RegionGrowing(cv::Mat image, int threeshold, int XPoint, int YPoint);
	void Agglomerative(cv::Mat image2);
	void AgglomerativeCall();
	void K_Means(cv::Mat image2);
	void K_MeansCall();
	void Mean_Shift(cv::Mat image2 , float s, float r);
	void Mean_ShiftCall();
	void _Otsu();
	void _OptimalThreshold();
	void _LocalThresholdOptimal();
	void _LocalThresholdOtsu();




};
#endif // MAINWINDOW_H
