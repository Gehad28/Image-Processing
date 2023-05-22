#include "dialog_edgdetection.h"
#include "ui_dialog_edgdetection.h"
#include "mainwindow.h"

Dialog_edgDetection::Dialog_edgDetection(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog_edgDetection)
{
    ui->setupUi(this);
//    cout<<cannyFlagDialogue<<endl;

    ui->sobelPrewittOptions->addItem("Ix");
    ui->sobelPrewittOptions->addItem("Iy");
    ui->sobelPrewittOptions->addItem("Ixy");

//    ui->sobelPrewittOptions->hide();
//    ui->label->hide();


}

Dialog_edgDetection::~Dialog_edgDetection()
{
    delete ui;
}

void Dialog_edgDetection::on_sobelPrewittOptions_activated(int index)
{
    sobel_prewitt_edgeOption = index;
}



