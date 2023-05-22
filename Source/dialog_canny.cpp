#include "dialog_canny.h"
#include "ui_dialog_canny.h"
#include "mainwindow.h"

Dialog_Canny::Dialog_Canny(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog_Canny)
{
    ui->setupUi(this);
    ui->lowThreshold->setValue(0.1);
    ui->highThreshold->setValue(0.3);
}

Dialog_Canny::~Dialog_Canny()
{
    delete ui;
}

void Dialog_Canny::on_buttonBox_accepted()
{
    lowThreshold = ui->lowThreshold->value();
    highThershold = ui->highThreshold->value();
    cout<<lowThreshold<<endl<<highThershold<<endl<<endl;
}

