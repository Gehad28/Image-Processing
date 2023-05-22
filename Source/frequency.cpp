#include "frequency.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <QComboBox>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;


Frequency::Frequency()
{

}

void expand_img_to_optimal(Mat& padded,Mat& img) {
    int row = getOptimalDFTSize(img.rows);
    int col = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, row - img.rows, 0, col - img.cols, BORDER_CONSTANT, Scalar::all(0));
}




Mat fourier_transform(Mat& img) {
    Mat padded;
    expand_img_to_optimal(padded, img);

    // Since the result of Fourier Transformation is in complex form we make two planes to hold  real and imaginary value
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI, DFT_COMPLEX_OUTPUT); // Fourier Transform

    return complexI;
}


void lowHighpassFilter( Mat& dft_filter, int distance, bool f)
{
    Mat tmp = Mat(dft_filter.rows, dft_filter.cols, CV_32F);
    Point center = Point(dft_filter.rows/2, dft_filter.cols/2);
    double radius;
    for(int i = 0; i < dft_filter.rows;++i) {
        for(int j=0; j < dft_filter.cols;++j) {
            radius = (double)(sqrt(pow((i-center.x), 2.0) + pow((j-center.y), 2.0)));
            if (f){
                if(radius < distance)
                    tmp.at<float>(i,j) = 0.0; //if point is in the radius make it zero
                else
                    tmp.at<float>(i,j) = 1.0;// Else make it one
            }
            if (!f){
                if(radius > distance)
                    tmp.at<float>(i,j) = 0.0; //if point is in the radius make it zero
                else
                    tmp.at<float>(i,j) = 1.0;// Else make it one
            }
        }
    }
    Mat toMerge[] = {tmp,tmp};
    merge(toMerge, 2, dft_filter); //since we are dealing with Two Channel image which is greyscale
}



void crop_and_rearrange(Mat& magI)
{
    if(magI.cols % 2 == 0 && magI.rows % 2 == 0)
        magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    else
        magI = magI(Rect(0, 0, magI.cols & -1, magI.rows & -1));
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

}



Mat_<float>MainWindow:: applyLowhighpassfilter(Mat& img, float radius, bool f){
    Mat complexI = fourier_transform(img);

    Mat filter = complexI.clone();
    lowHighpassFilter(filter, radius, f); //Our Low Pass Filter of Radius 30

    crop_and_rearrange(complexI);
    mulSpectrums(complexI, filter, complexI, 0); //Multiplying original image with filter image to get final image
    crop_and_rearrange(complexI);


    Mat planes[2], imgOutput;
    idft(complexI, complexI); //Reversing dft process to get our final image

    split(complexI, planes);
    normalize(planes[0], imgOutput, 0, 1, NORM_MINMAX);

    return imgOutput;
}

void MainWindow::hybridImage(Mat& img1, Mat& img2){
    Mat limg = applyLowhighpassfilter( img1, 30, false);
    imwrite(PATH + "Imglowpasshybrid.png", limg*255);
    Mat Imagelowpass = imread(PATH + "Imglowpasshybrid.png" , IMREAD_GRAYSCALE);

    Mat himg = applyLowhighpassfilter( img2, 30, true);
    cv::resize(himg,himg , Size(limg.cols,limg.rows));
    imwrite(PATH + "Imghighpasshybrid.png", himg*255);
    Mat Imagehighpass = imread(PATH + "Imghighpasshybrid.png" , IMREAD_GRAYSCALE);


    Mat hybridimg = Imagelowpass + Imagehighpass  ;
    displayImage(ui->hybridimage,hybridimg);

}
