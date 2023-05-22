#include "histogram.h"

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

Histogram::Histogram()
{

}


vector <vector <int >> calculateHist(Mat& image)

{
    Mat img = image.clone();
    vector <vector <int > > hist;
    if (img.channels() == 3){
        vector <int > row(256, 0);
        hist.resize(3, row);
        for (int i = 0; i < img.rows; i++) {

            for (int j = 0; j < img.cols; j++)
            {
                Vec3b bgr = img.at< Vec3b>(i, j);
                hist[0][(int)bgr[0]] += 1;
                hist[1][(int)bgr[1]] += 1;
                hist[2][(int)bgr[2]] += 1;
            }
        }
    }
    else {
        vector <int > row(256, 0);
        hist.resize(1, row);
        for (int i = 0; i < img.rows; i++) {

            for (int j = 0; j < img.cols; j++)
            {
                hist[0][(int)img.at<uchar>(i, j)] += 1;
            }
        }
    }
    return hist;
}


vector <int > accumulate( vector <int > arr)
{
    vector <int > acc_arr;
    acc_arr.push_back(arr[0]);
    for (int i = 1; i < arr.size(); i++)
    {
        acc_arr.push_back(arr[i]+ acc_arr[i-1]);
    }
    return acc_arr;
}


vector <int > normalizeHistogram(vector <int > arr,int height) /*normalize histogram values to fit in the plot image */ {
    auto max_value = *max_element(arr.begin(), arr.end());
    auto min_value = *min_element(arr.begin(), arr.end());
    vector <int > normalized_arr;
    for (int i = 0; i < arr.size(); i++)
    {
        normalized_arr.push_back(((arr[i] - min_value) * height / (max_value - min_value)));
    }

    return normalized_arr;
}

Mat3b DrawGreyHist(vector <int > hist, string name) {
    int binSize = 2;
    int  rows = 500;
    int cols = hist.size() * binSize;

    Mat3b plot = Mat3b(rows, cols, Vec3b(1, 1, 1));
    vector <int > histnorm = normalizeHistogram(hist, rows);

    for (int i = 0; i < hist.size(); ++i)
    {
        int h = rows - histnorm[i];
        rectangle(plot, Point(i * binSize, h), Point((i + 1) * binSize - 1, rows), (i % 2) ? Scalar(250, 0, 0) : Scalar(200, 0, 0), FILLED);
    }
    return plot;
}

Mat3b DrawGBRHist(vector <int > blue_hist, vector <int > green_hist ,vector <int > red_hist, string name) {

    int binSize = 2;
    int  rows = 500;
    int cols = blue_hist.size() * binSize;
    Mat3b plot = Mat3b(rows, cols, Vec3b(1, 1, 1));
    vector <int > blue=normalizeHistogram(blue_hist,rows);
    vector <int > green = normalizeHistogram(green_hist, rows);
    vector <int > red = normalizeHistogram(red_hist, rows);


    for (int i = 1; i < blue_hist.size(); i++)
    {
        line(plot, Point(binSize * (i - 1), rows - blue[i-1]),
            Point(binSize * (i), rows - blue[i]),
            Scalar(255, 0, 0), 2, 8, 0);
        line(plot, Point(binSize * (i - 1), rows - green[i-1]),
            Point(binSize * (i), rows - green[i]),
            Scalar(0, 255, 0), 2, 8, 0);
        line(plot, Point(binSize * (i - 1), rows - red[i-1]),
            Point(binSize * (i), rows - red[i]),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    return plot;
}

void MainWindow:: DrawHist(Mat& img, QLabel *hist_lbl, QLabel *cum) /*loops on pixels and count number of each value from 0 to 250 */
{
    if (img.channels() == 3) {
        vector <vector <int > > hist = calculateHist(img);
        blue_hist = hist[0];
        green_hist = hist[1];
        red_hist = hist[2];
        vector <int > acc_blue_hist = accumulate(blue_hist);
        vector <int > acc_green_hist = accumulate(green_hist);
        vector <int > acc_red_hist = accumulate(red_hist);
        Mat3b rgbHist = DrawGBRHist(blue_hist, green_hist, red_hist, "RGB Histogram");
        Mat3b rgbCum = DrawGBRHist(acc_blue_hist, acc_green_hist, acc_red_hist, "RGB Cumulative curve");
        displayImage(hist_lbl, rgbHist);
        displayImage(cum, rgbCum);
    }
    else {
        grey_hist = calculateHist(img)[0];
        vector <int > acc_grey_hist = accumulate(grey_hist);
        Mat3b grayHist = DrawGreyHist(grey_hist,"Grayscale Histogram");
        Mat3b grayCum = DrawGreyHist(acc_grey_hist,"Grayscale Cumulative curve");
        displayImage(hist_lbl, grayHist);
        displayImage(cum, grayCum);
    }
}


std::vector <float > mapping(std::vector <float > arr)
{
    std::vector <float > acc_arr;
    acc_arr.push_back(arr[0]);
    for (int i = 1; i < arr.size(); i++)
    {
        acc_arr.push_back(acc_arr[i - 1]+arr[i] );
    }

    for (int i = 1; i < arr.size(); i++)
    {
        acc_arr[i]=(floor(255 * acc_arr[i]));
    }
    return acc_arr;
}


std::vector <float > Equalize(std::vector <int > hist ,int numOfPixels) {


    std::vector <float > new_hist;

    for (int i = 0; i < hist.size(); i++)
    {

        float num = (float)hist[i] / (float)numOfPixels;
        new_hist.push_back( num);
    }
    std::vector <float > map= mapping(new_hist);
    return map;

}





Mat MainWindow::ImageEqualization(Mat& img) /*loops on pixels and count number of each value from 0 to 250 */
{
    DrawHist(img, ui->inputHist, ui->inputCum);
    if (img.empty())
    {
        Mat img2;
        img2 = Mat(200, 200, CV_8UC1, 1);
        return img2;
    }

    int numOfPixels = img.rows * img.cols;
    Mat img2;

    if (img.channels() == 3){
        img2 = Mat(img.rows, img.cols, CV_8UC3);
        std::vector <float > blueMap = Equalize(blue_hist, numOfPixels);
        std::vector <float > greenMap = Equalize(green_hist, numOfPixels);
        std::vector <float >  redMap = Equalize(red_hist, numOfPixels);


        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                Vec3b bgr = img.at< Vec3b>(i, j);

                img2.at< Vec3b>(i, j) = {
                    (unsigned char)blueMap[bgr[0]],
                     (unsigned char)greenMap[bgr[1]],
                     (unsigned char)redMap[bgr[2]]

                };
            }
        }
    }
    else {
        img2 = Mat(img.rows, img.cols, CV_8UC1);
        std::vector <float >  greyMap = Equalize(grey_hist, numOfPixels);


        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                uchar bgr = img.at< uchar>(i, j);
                img2.at< uchar>(i, j) = {
                    (unsigned char) greyMap[bgr],
                };
            }
        }
    }
    return img2;
}

Mat normalizeGrayImage(Mat& img,int minNum,int maxNum,Mat &normalizedImage){
    double minVal,maxVal;
    minMaxLoc(img,&minVal,&maxVal);
    for(int i =0;i < img.rows; i++){
        for(int j = 0 ; j<img.cols ; j++){
            uchar pixel = img.at<uchar>(i,j);
            normalizedImage.at<uchar>(i,j) = { (unsigned char)((((pixel - minVal)*(maxNum - minNum)) / (maxVal - minVal))+ minNum )};
        }
    }

return normalizedImage;
}

Mat MainWindow:: minMaxNormalization(Mat &img){
   int minNum = minMaxWindow.minNum;
    int maxNum = minMaxWindow.maxNum;
    double minVal,maxVal;
    minMaxLoc(img,&minVal,&maxVal);
    Mat normalizedImage = Mat(img.rows, img.cols, CV_8UC1, 255);
    if(grayFlag){
        normalizedImage = normalizeGrayImage(img,minNum,maxNum,normalizedImage);
    }
    else if(colorFlag){
        double minVals[3],maxVals[3];
        split(img,Bands);
        for(int i = 0;i<3;i++){
            minMaxLoc(img,&minVals[i],&maxVals[i]);
        }
        for(int i =0; i < 3; i++){
            Bands[i] =((((Bands[i] - minVals[i])*(maxNum - minNum)) / (maxVals[i] - minVals[i]))+ minNum );

        }
        merge(Bands,3,normalizedImage);

    }

    return normalizedImage;
}

Mat globalThresholdLoop(Mat& inputImg, Mat& outputImg, double thresh,int x){
    for(int i =0;i < inputImg.rows; i++){
        for(int j = 0 ; j<inputImg.cols ; j++){
            double pixel = inputImg.at<uchar>(i,j);
            uchar newPixel = 0;

            if(pixel > thresh){
                newPixel = 255;
            }
            if(x == 1){

            outputImg.at<uchar>(i,j) = newPixel;
            }
            else{
                outputImg.at<Vec3b>(i,j) = Vec3b(newPixel,newPixel,newPixel);
            }
        }
    }
    return outputImg;

}

void MainWindow:: globalThreshold(Mat& inputImg, Mat& outputImg, double thresh){
    if(grayFlag){
        outputImg = globalThresholdLoop(inputImg,outputImg,thresh,1);
    }
    if(colorFlag){
        outputImg = inputImg.clone();
        Mat grayImage;
        grayImage =  Grayscale(inputImg);
        outputImg = globalThresholdLoop(grayImage,outputImg,thresh,2);

    }
}

Mat adaptiveLocalLoop(Mat& inputImg,Mat& outputImg,int blockSize, int constant,Mat &paddedImg,Mat &meanImg,int x){
    for(int i = 0 ; i < inputImg.rows ; i++){
        for (int j = 0 ; j < inputImg.cols ; j ++){
            float sum = 0;
            for(int m = -blockSize/2; m <= blockSize/2; m++){
                for(int n = -blockSize/2; n <= blockSize/2 ; n++){
                    sum += paddedImg.at<uchar>(i+m+blockSize/2,j+n+blockSize/2);
                }

            }
                meanImg.at<float>(i,j) = sum / (blockSize*blockSize);
        }
    }
    if(x == 1){
        for(int i =0 ; i< inputImg.rows; i++){
            for(int j = 0 ; j< inputImg.cols; j++){
                int thresholdVal = (int)meanImg.at<float>(i,j) - constant;
                if(inputImg.at<uchar>(i,j)>= thresholdVal){
                    outputImg.at<uchar>(i,j) = 255;
                }
            }

        }

    }
    else{
        for(int i =0 ; i< inputImg.rows; i++){
            for(int j = 0 ; j< inputImg.cols; j++){
                int thresholdVal = (int)meanImg.at<float>(i,j) - constant;
                if(inputImg.at<uchar>(i,j)>= thresholdVal){
                    outputImg.at<Vec3b>(i,j) = Vec3b(255,255,255);
                }
            }

        }
    }
    return outputImg;

}

void MainWindow::adaptiveLocalThreshold(Mat& inputImg,Mat& outputImg,int blockSize, int constant){
    Mat meanImg = Mat::zeros(inputImg.rows,inputImg.cols,CV_32F);
    Mat paddedImg,greyImage;
    copyMakeBorder(inputImg,paddedImg,blockSize/2,blockSize/2,blockSize/2,blockSize/2,BORDER_CONSTANT);
    outputImg = Mat::zeros(inputImg.rows,inputImg.cols,CV_8U);
    outputImg = adaptiveLocalLoop(inputImg,outputImg,blockSize,constant,paddedImg,meanImg,1);
//    if(grayFlag){
//        outputImg = adaptiveLocalLoop(inputImg,outputImg,blockSize,constant,paddedImg,meanImg,1);
//    }
//    if(colorFlag){
//        Mat grayImage ;
//        grayImage = Grayscale(inputImg);
//        Mat meanImage = Mat::zeros(grayImage.rows,grayImage.cols,CV_8UC1);
//        copyMakeBorder(grayImage,paddedImg,blockSize/2,blockSize/2,blockSize/2,blockSize/2,BORDER_REPLICATE);
//        outputImg = Mat::zeros(inputImg.rows,inputImg.cols,CV_8UC3);
//        outputImg = adaptiveLocalLoop(grayImage,outputImg,11,2,paddedImg,meanImg,2);

//    }



}



