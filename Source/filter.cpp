#include "filter.h"
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

filter::filter()
{

}


//__________________Noise______________________

Mat addUniformNoise(const Mat& img, float amplitude) {
    // Create a random number generator
    default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-amplitude, amplitude);

    // Create a noisy copy of the input image
    Mat noisyImg = img.clone();

    // Add uniform noise to the image
    for (int i = 0; i < noisyImg.rows; ++i) {
        for (int j = 0; j < noisyImg.cols; ++j) {
            float noise = distribution(generator);
            noisyImg.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(noisyImg.at<Vec3b>(i, j)[0] + noise);
            noisyImg.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(noisyImg.at<Vec3b>(i, j)[1] + noise);
            noisyImg.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(noisyImg.at<Vec3b>(i, j)[2] + noise);
        }
    }

    return noisyImg;
}



Mat addGaussianNoise(Mat& img, double mean, double stddev)
{
    // Create a random number generator
    default_random_engine generator;
    normal_distribution<double> distribution(mean, stddev);

    Mat noisyImg = img.clone();

    // Loop through each pixel of the img and add Gaussian noise
    for (int y = 0; y < noisyImg.rows; y++)
    {
        for (int x = 0; x < noisyImg.cols; x++)
        {
            // Get the color channels of the pixel
            Vec3b& pixel = noisyImg.at<Vec3b>(y, x);
            double r = static_cast<double>(pixel[0]);
            double g = static_cast<double>(pixel[1]);
            double b = static_cast<double>(pixel[2]);

            // Add Gaussian noise to each color channel
            r += distribution(generator);
            g += distribution(generator);
            b += distribution(generator);

            // Clamp the color values to the range [0, 255]
            r = max(0.0, min(255.0, r));
            g = max(0.0, min(255.0, g));
            b = max(0.0, min(255.0, b));

            // Set the color channels of the pixel
            pixel[0] = static_cast<uint8_t>(r);
            pixel[1] = static_cast<uint8_t>(g);
            pixel[2] = static_cast<uint8_t>(b);
        }
    }
    return noisyImg;
}



Mat addSaltAndPepperNoise(Mat& img, double noise_ratio)
{
    int height = img.rows;
    int width = img.cols;
    int total_pixels = height * width;

    int num_salt_pixels = static_cast<int>(total_pixels * noise_ratio);
    int num_pepper_pixels = static_cast<int>(total_pixels * noise_ratio);

    default_random_engine generator;
    uniform_int_distribution<int> salt_distribution(0, height - 1);
    uniform_int_distribution<int> pepper_distribution(0, width - 1);

    Mat noisyImg = img.clone();

    for (int i = 0; i < num_salt_pixels; i++)
    {
        int x = salt_distribution(generator);
        int y = pepper_distribution(generator);
        noisyImg.at<Vec3b>(x, y) = Vec3b(255, 255, 255);
    }

    for (int i = 0; i < num_pepper_pixels; i++)
    {
        int x = salt_distribution(generator);
        int y = pepper_distribution(generator);
        noisyImg.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
    }
    return noisyImg;
}


Mat MainWindow::applyNoise(Mat &image){
    Mat img2 = image;
    if (grayFlag)
        img2 = filteredImage;
    if(ui->noise->currentText() == "Uniform Noise"){
        Mat outImageUniform =addUniformNoise(img2, noisewindow.percent) ;
        if (grayFlag){
            outImageUniform =Grayscale(outImageUniform);
        }
        displayImage(ui->outputImage, outImageUniform);
        return outImageUniform;
    }


    if(ui->noise->currentText() == "Gaussian Noise"){
        Mat outImageGnoise =addGaussianNoise(img2, noisewindow2.mean, noisewindow2.std) ;
        if (grayFlag){
            outImageGnoise =Grayscale(outImageGnoise);
        }
        displayImage(ui->outputImage, outImageGnoise);
        return outImageGnoise;
    }

    if(ui->noise->currentText() == "Salt and Pepper Noise"){
        Mat outImageSPnoise =addSaltAndPepperNoise( img2, noisewindow.percent/100) ;
        if (grayFlag){
            outImageSPnoise = Grayscale(outImageSPnoise);
        }
        displayImage(ui->outputImage, outImageSPnoise);
        return outImageSPnoise;
    }

    if(ui->noise->currentText() == "None"){
        displayImage(ui->outputImage, this->image);
        return this->image;
    }
}



//________________Bluring Filters___________________

Mat_<float> gaussianFilter(int n)
{
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    double sum = 0.0;

    Mat_<float> gaussMatrix(n, n);

    for (int x = -(n/2); x <= (n/2); x++) {
        for (int y = -(n/2); y <= (n/2); y++) {
            r = sqrt(x * x + y * y);
            gaussMatrix.at<float>(x + (n/2), y + (n/2)) = (exp(-(r * r) / s)) / (2*M_PI * s);
            sum += gaussMatrix.at<float>(x + (n/2), y + (n/2));
        }
    }

    gaussMatrix = gaussMatrix / sum;
    return gaussMatrix.clone();
}




Mat averageFilter(int n){
    Mat averageMatrix = Mat(n, n, CV_32FC1, Scalar(1.0));
    averageMatrix = averageMatrix / (n * n);
    return averageMatrix;
}




Mat_<float> convolution(const Mat_<float>& img, const Mat_<float>& filter)
{
    Mat dst(img.rows,img.cols,img.type());

    const int dx = filter.cols / 2;
    const int dy = filter.rows / 2;

    for (int i = 0; i<img.rows; i++)
    {
        for (int j = 0; j<img.cols; j++)
        {
            float tmp = 0.0f;
            for (int k = 0; k<filter.rows; k++)
            {
              for (int l = 0; l<filter.cols; l++)
              {
                int x = j - dx + l;
                int y = i - dy + k;
                if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
                    tmp += img.at<float>(y, x) * filter.at<float>(k, l);
              }
            }
            dst.at<float>(i, j) = saturate_cast<float>(tmp);
        }
    }
    return dst.clone();
}




float getMedian(const Mat_<float>& matrix, int n){
    int temp[n*n], s = 0;
    float median = 0.0;
    //copying all element of matrix in temp
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++, s++)
            temp[s] = matrix.at<float>(i, j);

    //sorting temp array
    sort(temp, temp + s);
    median = temp[n / 2];
    return median;
}




Mat_<float> medianFilter(const Mat_<float>& img, int n)
{
    Mat_<float> filter(n, n);
    Mat dst(img.rows,img.cols,img.type());
    float median = 0;

    const int dx = filter.cols / 2;
    const int dy = filter.rows / 2;

    for (int i = 0; i<img.rows; i++)
    {
        for (int j = 0; j<img.cols; j++)
        {
            for (int k = 0; k<filter.rows; k++)
            {
              for (int l = 0; l<filter.cols; l++)
              {
                int x = j - dx + l;
                int y = i - dy + k;
                if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
                    filter.at<float>(k, l) = img.at<float>(y, x);
              }
            }
            median = getMedian(filter, n);
            dst.at<float>(i, j) = median;
        }
    }
    return dst.clone();
}


Mat MainWindow::applyFilter(Mat &image){
    if(ui->filterList->currentText() == "Gaussian Filter"){
        if(grayFlag){
            gaussFilter = gaussianFilter(popUpwindow.choise);
            Mat outImageGauss = convolution(image, gaussFilter);
            imwrite(PATH + "outputImgGauss.png", outImageGauss);
            outImageGauss = imread(PATH + "outputImgGauss.png" , IMREAD_GRAYSCALE);
            displayImage(ui->outputImage, outImageGauss);
            return outImageGauss;
        }

        if(colorFlag){
            split(image, Bands);
            gaussFilter = gaussianFilter(popUpwindow.choise);
            Mat outImageGaussCH0 = convolution(Bands[0], gaussFilter); //blue
            Mat outImageGaussCH1 = convolution(Bands[1], gaussFilter);  //green
            Mat outImageGaussCH2 = convolution(Bands[2], gaussFilter); //red
            channels = {outImageGaussCH0,outImageGaussCH1,outImageGaussCH2};
            merge(channels,merged);
            imwrite(PATH + "outputImgGausscolor.png", merged);
            Mat outImageGausscolor = imread(PATH + "outputImgGausscolor.png" , 1);
            displayImage(ui->outputImage, outImageGausscolor);
            return outImageGausscolor;
        }
    }

    if(ui->filterList->currentText() == "Average Filter"){
        if(grayFlag){
            avgFilter = averageFilter(popUpwindow.choise);
            Mat outImageAvg = convolution(image, avgFilter);
            imwrite(PATH + "outputImgAvg.png", outImageAvg);
            outImageAvg = imread(PATH + "outputImgAvg.png" , IMREAD_GRAYSCALE);
            displayImage(ui->outputImage, outImageAvg);
            return outImageAvg;
        }
        if(colorFlag){
            split(image, Bands);
            avgFilter = averageFilter(popUpwindow.choise);
            Mat outImageAvgCH0 = convolution(Bands[0], avgFilter);
            Mat outImageAvgCH1 = convolution(Bands[1], avgFilter);
            Mat outImageAvgCH2 = convolution(Bands[2], avgFilter);
            channels = {outImageAvgCH0,outImageAvgCH1,outImageAvgCH2};
            merge(channels,merged);
            imwrite(PATH + "outputImgAvgcolor.png", merged);
            Mat outImageAvgcolor = imread(PATH + "outputImgAvgcolor.png" , 1);
            displayImage(ui->outputImage, outImageAvgcolor);
            return outImageAvgcolor;
        }
    }

    if(ui->filterList->currentText() == "Median Filter"){
        if(grayFlag){
            Mat outImageMedian = medianFilter(image, popUpwindow.choise);
            imwrite(PATH + "outputImgMedian.png", outImageMedian);
            outImageMedian = imread(PATH + "outputImgMedian.png" , IMREAD_GRAYSCALE);
            displayImage(ui->outputImage, outImageMedian);
            return outImageMedian;
        }
        if(colorFlag){
            split(image, Bands);
            Mat outImageMedianCH0 = medianFilter(Bands[0], popUpwindow.choise);
            Mat outImageMedianCH1 = medianFilter(Bands[1], popUpwindow.choise);
            Mat outImageMedianCH2 = medianFilter(Bands[2], popUpwindow.choise);
            channels = {outImageMedianCH0,outImageMedianCH1,outImageMedianCH2};
            merge(channels,merged);
            imwrite(PATH + "outputImgMediancolor.png", merged);
            Mat outImageMediancolor = imread(PATH + "outputImgMediancolor.png" , 1);
            displayImage(ui->outputImage, outImageMediancolor);
            return outImageMediancolor;
        }
    }

    if(ui->filterList->currentText() == "Lowpass Filter"){
        if(grayFlag){
            Mat outImagelowpass = applyLowhighpassfilter(image, ftWindow.radius, true);
            imwrite(PATH + "outputImglowpass.png", outImagelowpass*255);
            outImagelowpass = imread(PATH + "outputImglowpass.png" , IMREAD_GRAYSCALE);
            displayImage(ui->outputImage, outImagelowpass);
            return outImagelowpass;
        }
        if(colorFlag){
            split(image, Bands);
            Mat outImagelowpassCH0 = applyLowhighpassfilter(Bands[0], ftWindow.radius, true);
            Mat outImagelowpassCH1 = applyLowhighpassfilter(Bands[1], ftWindow.radius, true);
            Mat outImagelowpassCH2 = applyLowhighpassfilter(Bands[2], ftWindow.radius, true);
            channels = {outImagelowpassCH0,outImagelowpassCH1,outImagelowpassCH2};
            merge(channels, merged);
            imwrite(PATH + "outputImglowpasscolor1.png", merged*255);
            Mat outImagelowpasscolor = imread(PATH + "outputImglowpasscolor1.png" , 1);
            displayImage(ui->outputImage, outImagelowpasscolor);
            return outImagelowpasscolor;
        }
    }

    if(ui->filterList->currentText() == "Highpass Filter"){
        if(grayFlag){
            Mat outImagehighpass = applyLowhighpassfilter(image, ftWindow.radius, false);
            imwrite(PATH + "outputImghighpass.png", outImagehighpass*255);
            outImagehighpass = imread(PATH + "outputImghighpass.png" , IMREAD_GRAYSCALE);
            displayImage(ui->outputImage, outImagehighpass);
            return outImagehighpass;
        }
        if(colorFlag){
            split(image, Bands);
            Mat outImagehighpassCH0 = applyLowhighpassfilter(Bands[0], ftWindow.radius, false);
            Mat outImagehighpassCH1 = applyLowhighpassfilter(Bands[1], ftWindow.radius, false);
            Mat outImagehighpassCH2 = applyLowhighpassfilter(Bands[2], ftWindow.radius, false);
            channels = {outImagehighpassCH0,outImagehighpassCH1,outImagehighpassCH2};
            merge(channels, merged);
            imwrite(PATH + "outputImghighpasscolor.png", merged*255);
            Mat outImagehighpasscolor = imread(PATH + "outputImghighpasscolor.png" , 1);
            displayImage(ui->outputImage, outImagehighpasscolor);
            return outImagehighpasscolor;
        }
    }
    if(ui->filterList->currentText() == "None"){
        displayImage(ui->outputImage, this->image);
        return this->image;
    }
}




//_______________Edge Detection_________________


Mat MainWindow::roberts_EdgeDetector(Mat& src){
    Mat vertical = (Mat_<float>(2,2) <<
                    1, 0,
                    0, -1);

    Mat horizontal = (Mat_<float>(2,2)<<
                      0, 1,
                      -1, 0);

    Mat vEdges, hEdges;
    vEdges = convolution(src, vertical);
    hEdges = convolution(src, horizontal);

    Mat outImage;
    magnitude(vEdges, hEdges, outImage);

    return outImage;
}

Mat MainWindow::sobel_prewitt_EdgeDetector(Mat& src,int sobel_prewitt_Flag, int edgeOption){
    // sobel_prewitt_Flag: a flag to determine weather the user chose prewitt or sobel
    // edgeOption an int to determine which edges (Ix, Iy, or Ixy) the user chose
    Mat kernelX, kernelY;
    if (sobel_prewitt_Flag == 0){               //Prewitt
        kernelX = (Mat_<float>(3, 3) <<
                       -1, 0, 1,
                       -1, 0, 1,
                       -1, 0, 1);
        kernelY = (Mat_<float>(3, 3) <<
                       -1, -1, -1,
                        0,  0,  0,
                        1,  1,  1);

    }
    else{                                       //Sobel
        kernelX = (Mat_<float>(3, 3) <<
                       -1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1);
        kernelY = (Mat_<float>(3, 3) <<
                       -1, -2, -1,
                        0,  0,  0,
                        1,  2,  1);
    }
    // Convolve image with kernels
    Mat xEdges, yEdges;
    xEdges = convolution(src, kernelX);
    yEdges = convolution(src, kernelY);

    if (edgeOption == 0){
        return xEdges;
    }
    else if (edgeOption == 1){
        return yEdges;
    }
    else{
        //Sum the 2 edges
        Mat xyEdges;
        addWeighted(abs(xEdges), 1, abs(yEdges), 1, 0, xyEdges);
        return xyEdges;
    }
}

Mat nonMaxSuppression(const Mat &magnitude, const Mat &direction)
{
    Mat suppressedImage(magnitude.rows, magnitude.cols, CV_32F, Scalar(0));
    float angle, a, b, c;

    for (int i = 1; i < magnitude.rows - 1; i++)
    {
        for (int j = 1; j < magnitude.cols - 1; j++)
        {
            angle = direction.at<float>(i, j);

            if ((angle < 22.5 && angle >= 0) || (angle >= 157.5 && angle < 202.5) || (angle >= 337.5 && angle <= 360))
            {
                a = magnitude.at<float>(i, j - 1);
                b = magnitude.at<float>(i, j + 1);
            }
            else if ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5))
            {
                a = magnitude.at<float>(i - 1, j + 1);
                b = magnitude.at<float>(i + 1, j - 1);
            }
            else if ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5))
            {
                a = magnitude.at<float>(i - 1, j);
                b = magnitude.at<float>(i + 1, j);
            }
            else
            {
                a = magnitude.at<float>(i - 1, j - 1);
                b = magnitude.at<float>(i + 1, j + 1);
            }

            c = magnitude.at<float>(i, j);
            if (c > a && c > b)
            {
                suppressedImage.at<float>(i, j) = c;
            }
        }
    }

    return suppressedImage;
}

Mat doubleThreshold(Mat &image, double lowThreshold, double highThreshold)
{
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            if (image.at<float>(i, j) < lowThreshold)
            {
                image.at<float>(i, j) = 0;
            }
            else if (image.at<float>(i, j) > highThreshold)
            {
                image.at<float>(i, j) = 255;
            }
            else
            {
                image.at<float>(i, j) = 128;
            }
        }
    }
    return image;
}

Mat Normalize(Mat&src){
    double minX, maxX;
    minMaxLoc(src, &minX, &maxX);
    float max = static_cast<float>(maxX);
    cout<<max<<endl;
    Mat normalizedMatrix = src / max;
    return normalizedMatrix;
}

Mat MainWindow::canny_EdgeDetector(Mat &src, double lowThreshold, double highThreshold)
{
    // Convert the input image to grayscale
//    if (colorFlag){
//        src = Grayscale(src);
//    }


    Mat gaussFilter = gaussianFilter(3);
    Mat smoothed_image = convolution(grayImage, gaussFilter);
    Mat grad_x = sobel_prewitt_EdgeDetector(smoothed_image, 1, 0);
    Mat grad_y = sobel_prewitt_EdgeDetector(smoothed_image, 1, 1);
    grad_x = Normalize(grad_x);
    grad_y = Normalize(grad_y);

    Mat magnitude, direction;
    cartToPolar(grad_x, grad_y, magnitude, direction, true);

    // Apply non-maximum suppression
    Mat suppressed_image = nonMaxSuppression(magnitude, direction);

    // Apply double thresholding
    doubleThreshold(suppressed_image, lowThreshold, highThreshold);

    // Apply edge tracking by hysteresis
    Mat output_image = Mat::zeros(suppressed_image.rows, suppressed_image.cols, CV_8UC1);
    for (int i = 0; i < suppressed_image.rows; ++i)
    {
        for (int j = 0; j < suppressed_image.cols; ++j)
        {
            if (suppressed_image.at<float>(i, j) == 255)
            {
                // Start tracking the edge
                output_image.at<uchar>(i, j) = 255;

                // Check neighbors to see if they are also edges
                for (int ii = i - 1; ii <= i + 1; ++ii)
                {
                    for (int jj = j - 1; jj <= j + 1; ++jj)
                    {
                        if (ii >= 0 && ii < suppressed_image.rows && jj >= 0 && jj < suppressed_image.cols)
                        {
                            if (suppressed_image.at<float>(ii, jj) == 128)
                            {
                                // Mark this neighbor as an edge and continue tracking
                                suppressed_image.at<float>(ii, jj) = 255;
                                output_image.at<uchar>(ii, jj) = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    return output_image;
}


Mat MainWindow::edgeDetection(Mat& src,int filterChoice, int edgeOption){
    Mat outImage;
    if (filterChoice == 0){        //Prewitt
        outImage = sobel_prewitt_EdgeDetector(src, filterChoice, edgeOption);
    }
    else if (filterChoice == 1){   //Sobel
        outImage = sobel_prewitt_EdgeDetector(src, filterChoice, edgeOption);
    }
    else if (filterChoice == 2){   //Roberts
        outImage = roberts_EdgeDetector(src);
    }
    else if (filterChoice == 3){   //Canny
        outImage = canny_EdgeDetector(src, cannyWindow.lowThreshold, cannyWindow.highThershold);
    }
    return outImage;
}

void MainWindow::applyEdgedetection(){
    int filterIndex = 0;
//    edgWindow.cannyFlagDialogue = false;

    if(ui->edgeDetection->currentText() == "Prewitt"){
        filterIndex = 0;
//        cannyFlag = false;
//        ui->sobelPrewittOptions->show();
    }
    else if (ui->edgeDetection->currentText() == "Sobel"){
        filterIndex = 1;
//        cannyFlag = false;
//        ui->sobelPrewittOptions->show();
    }
    else if (ui->edgeDetection->currentText() == "Roberts"){
        filterIndex = 2;
//        cannyFlag = false;
    }
    else if (ui->edgeDetection->currentText() == "Canny"){
        filterIndex = 3;
//        cannyFlag = true;
    }
    else if(ui->edgeDetection->currentText() == "None"){
//        cannyFlag = false;
        displayImage(ui->outputImage, image);
//        edgWindow.cannyFlagDialogue = true;
    }

//     edgWindow.show_hide(cannyFlag);

//    filters filter;
    Mat outImage;
    if (colorFlag){
//        if (cannyFlag){
//            outImage = edgeDetection(image, filterIndex, edgWindow.sobel_prewitt_edgeOption);
//        }
//        else{
        split(image, Bands);
        Mat outImage1 = edgeDetection(Bands[0], filterIndex, edgWindow.sobel_prewitt_edgeOption);
        Mat outImage2 = edgeDetection(Bands[1], filterIndex, edgWindow.sobel_prewitt_edgeOption);
        Mat outImage3 = edgeDetection(Bands[2], filterIndex, edgWindow.sobel_prewitt_edgeOption);
        channels = {outImage1, outImage2, outImage3};
        merge(channels,outImage);
//        }

    }
    else{
        outImage = edgeDetection(image, filterIndex, edgWindow.sobel_prewitt_edgeOption);
    }
    imwrite(PATH + "outImageEdge.png", outImage);
    outImage = imread(PATH + "outImageEdge.png" , IMREAD_COLOR);
    displayImage(ui->outputImage, outImage);
}
