#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <QComboBox>
#include "dialog.h"
#include "dialog2.h"
#include "dialog_edgdetection.h"
#include "dialognois.h"
#include "dialognoise.h"
#include "dialogminmax.h"
#include "dialogglobalthreshold.h"
#include "dialog_canny.h"

using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void displayImage(QLabel *img_lbl, Mat img);
    void readImage(QLabel *img_lbl, QComboBox *list);
    Mat applyNoise(Mat &image);
    Mat applyFilter(Mat &image);
    void showWindow(int x);
    Mat_<float> applyLowhighpassfilter(cv::Mat& img, float radius, bool f);

    Mat roberts_EdgeDetector(Mat& src);
    Mat sobel_prewitt_EdgeDetector(Mat& src,int sobel_prewitt_Flag, int edgeOption);
    Mat canny_EdgeDetector(Mat &gray_image, double lowThreshold, double highThreshold);

    Mat edgeDetection(Mat& src,int filterChoice, int edgeOption);
    void applyEdgedetection();

    Mat Grayscale(Mat& img);
    Dialog_edgDetection edgWindow;

    Mat minMaxNormalization(Mat &image);
    void globalThreshold(Mat &inputImg, Mat &outputImg, double thresh);
    void adaptiveLocalThreshold(Mat& inputImg, Mat& outputImg,int blockSize, int constant);


    void hybridImage(Mat& img_1, Mat& img_2);

    void DrawHist(Mat& img, QLabel *hist, QLabel *cum);

    Mat image;
    Mat grayImage;
    Mat imageRGB;
    Mat outImage;
    Mat imgl;
    Mat filteredImage;
    Mat normImage;

    int w = 400;
    int h = 500;
    QString file_name;
    vector<Mat> channels;
    Mat Bands[3], merged;
    Mat_<float> gaussFilter;
    Mat_<float> avgFilter;
    bool colorFlag = false;
    bool grayFlag = false;
    bool cannyFlag = false;
    String PATH = "..\\savedImages\\";
    Dialog popUpwindow;
    Dialog2 ftWindow;
    Dialognois noisewindow;
    Dialognoise noisewindow2;

    DialogMinMax minMaxWindow;
    DialogGlobalThreshold globalThresholdWindow;
    Dialog_Canny cannyWindow;




    vector <int > blue_hist = vector (256,1);
    vector <int > green_hist = vector(256, 1);
    vector <int > red_hist = vector (256, 1);
    vector <int > grey_hist = vector(256, 1);

    Mat ImageEqualization(Mat& img);

private slots:
    void on_Browse_clicked();

    void on_colorgrayList_activated();

    void on_filterList_activated();


    void on_Browse2_clicked();

    void on_colorgrayList2_activated();

    void on_noise_activated();

    void on_edgeDetection_activated();

    void on_histogram_clicked();

    void on_equalize_clicked();

    void on_normalizeButton_clicked();

    void on_globalThresholdButton_clicked();

    void on_Browse_2_clicked();

    void on_Browse_3_clicked();

    void on_localThresold_clicked();

    void on_Browse4_clicked();

    void on_reset_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
