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
#include "dialog.h"

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->colorgrayList->addItem("Color");
    ui->colorgrayList->addItem("Gray");
    ui->colorgrayList2->addItem("Color");
    ui->colorgrayList2->addItem("Gray");

    ui->filterList->addItem("None");
    ui->filterList->addItem("Gaussian Filter");
    ui->filterList->addItem("Average Filter");
    ui->filterList->addItem("Median Filter");
    ui->filterList->addItem("Lowpass Filter");
    ui->filterList->addItem("Highpass Filter");

    ui->noise->addItem("None");
    ui->noise->addItem("Uniform Noise");
    ui->noise->addItem("Gaussian Noise");
    ui->noise->addItem("Salt and Pepper Noise");

    ui->edgeDetection->addItem("None");
    ui->edgeDetection->addItem("Prewitt");
    ui->edgeDetection->addItem("Sobel");
    ui->edgeDetection->addItem("Roberts");
    ui->edgeDetection->addItem("Canny");

}

MainWindow::~MainWindow()
{
    delete ui;
}


//___________________Displaying__________________________


Mat MainWindow::Grayscale(Mat& img)
{
    Mat img2;
    img2 = Mat(img.rows, img.cols, CV_8UC1, 255);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 1; j < img.cols; j++)
        {
            Vec3b bgr = img.at< Vec3b>(i, j);

            img2.at< uchar>(i, j) = {
               (unsigned char) ((bgr[0] + bgr[1] + bgr[2]) / 3)
            };
        }
    }
    return img2;
}

void MainWindow::readImage(QLabel *img_lbl, QComboBox *list){
    grayImage = imread(file_name.toStdString() , 1);
    grayImage = Grayscale(grayImage);
    if(list->currentText() == "Color"){
        colorFlag = true;
        grayFlag = false;
        image = imread(file_name.toStdString() , 1);
        displayImage(img_lbl, image);
    }

    if(list->currentText() == "Gray"){
        colorFlag = false;
        grayFlag = true;
        image = imread(file_name.toStdString() , 1);
        image = Grayscale(image);
        displayImage(img_lbl, image);
    }
}


void MainWindow::displayImage(QLabel *img_lbl, Mat img){
    cvtColor(img, imageRGB, COLOR_BGR2RGB);
    img_lbl->setPixmap(QPixmap::fromImage(QImage(imageRGB.data, imageRGB.cols, imageRGB.rows, imageRGB.step,
                                                      QImage::Format_RGB888)).scaled(w,h,Qt::KeepAspectRatio));
    img_lbl->repaint();
}


void MainWindow::showWindow(int x){
    if (x == 1){
        if(ui->filterList->currentText() == "Gaussian Filter" || ui->filterList->currentText() == "Median Filter" ||
                ui->filterList->currentText() == "Average Filter"){
            popUpwindow.setModal(true);
            popUpwindow.exec();
        }

        if(ui->filterList->currentText() == "Lowpass Filter" || ui->filterList->currentText() == "Highpass Filter"){
            ftWindow.setModal(true);
            ftWindow.exec();
        }
    }

    if (x == 2){
        if(ui->edgeDetection->currentText() == "Prewitt" || ui->edgeDetection->currentText() == "Sobel"){
            edgWindow.setModal(true);
            edgWindow.exec();
        }
        else if (ui->edgeDetection->currentText() == "Canny"){
            cannyWindow.setModal(true);
            cannyWindow.exec();
        }
    }

    if (x == 3){
        if(ui->noise->currentText() == "Gaussian Noise"){
            noisewindow2.setModal(true);
            noisewindow2.exec();
        }
        if(ui->noise->currentText() == "Uniform Noise" || ui->noise->currentText() == "Salt and Pepper Noise"){
            noisewindow.setModal(true);
            noisewindow.exec();
        }
    }

    if (x == 4){
            minMaxWindow.setModal(true);
            minMaxWindow.exec();

        }
    if (x == 5){
        globalThresholdWindow.setModal(true);
        globalThresholdWindow.exec();
    }
//    if (x == 6){
//        cannyWindow.setModal(true);
//        cannyWindow.exec();
//    }
}



//____________TAB 1________________


void MainWindow::on_Browse_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if (!file_name.isEmpty()){
        readImage(ui->inputImage, ui->colorgrayList);
        displayImage(ui->outputImage, image);
        outImage = image;
    }
}


void MainWindow::on_colorgrayList_activated()
{
    if (!file_name.isEmpty()){
        readImage(ui->inputImage, ui->colorgrayList);
        displayImage(ui->outputImage, image);
    }
}

void MainWindow::on_noise_activated()
{
    if (!file_name.isEmpty()){
        showWindow(3);
        if(colorFlag)
            filteredImage = outImage;
        outImage = applyNoise(outImage);
    }
}

void MainWindow::on_filterList_activated()
{
    if (!file_name.isEmpty()){
        showWindow(1);
        outImage = applyFilter(outImage);
    }
}

void MainWindow::on_edgeDetection_activated()
{
    if (!file_name.isEmpty()){
        showWindow(2);
        applyEdgedetection();
    }
}

void MainWindow::on_reset_clicked()
{
    outImage = image;
    displayImage(ui->outputImage, image);
}



//______________TAB 2__________________


void MainWindow::on_Browse2_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if (!file_name.isEmpty()){
        readImage(ui->inputImage2, ui->colorgrayList2);
    }
}


void MainWindow::on_colorgrayList2_activated()
{
    if (!file_name.isEmpty()){
        readImage(ui->inputImage2, ui->colorgrayList2);
    }
}


void MainWindow::on_histogram_clicked()
{
    DrawHist(image, ui->inputHist, ui->inputCum);
}


void MainWindow::on_equalize_clicked()
{
    Mat img = ImageEqualization(image);
    displayImage(ui->outputImage2, img);
    DrawHist(img, ui->equalized, ui->equalizedCum);
}



//_______________TAB 3___________________

void MainWindow::on_Browse4_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if (!file_name.isEmpty()){
        normImage = imread(file_name.toStdString() , 1);
        image = Grayscale(normImage);
        displayImage(ui->inputImage4, image);
    }

}


void MainWindow::on_normalizeButton_clicked()
{
    showWindow(4);
    Mat normalizedImage = minMaxNormalization(image);
    displayImage(ui->outputImage4, normalizedImage);

}

void MainWindow::on_globalThresholdButton_clicked()
{
    showWindow(5);
    Mat outputImg = Mat(image.rows,image.cols,CV_8UC1);
    globalThreshold(image,outputImg,globalThresholdWindow.thresholdVal);

    displayImage(ui->outputImage4,outputImg);

}

void MainWindow::on_localThresold_clicked()
{
    Mat outputImg = Mat(image.rows,image.cols,CV_8UC1);
//    globalThreshold(image,outputImg,globalThresholdWindow.thresholdVal,globalThresholdWindow.maxInput);
    adaptiveLocalThreshold(image, outputImg, 11, 2);
    displayImage(ui->outputImage4,outputImg);
}



//_______________TAB 4____________________


void MainWindow::on_Browse_2_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if (!file_name.isEmpty()){
        imgl = imread(file_name.toStdString() , IMREAD_GRAYSCALE);
        displayImage(ui->inputImage_2, imgl);
    }
}

void MainWindow::on_Browse_3_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if (!file_name.isEmpty()){
        Mat img = imread(file_name.toStdString() , IMREAD_GRAYSCALE);
        displayImage(ui->inputImage_3, img);
        hybridImage(img, imgl);
    }

}









