#include "dialogminmax.h"
#include "ui_dialogminmax.h"

DialogMinMax::DialogMinMax(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogMinMax)
{
    ui->setupUi(this);
}

DialogMinMax::~DialogMinMax()
{
    delete ui;
}

void DialogMinMax::on_minNum_textChanged()
{
//    minNum = ui->minNum->value();
    minNum = ui->minNum->value();


}


void DialogMinMax::on_maxNum_textChanged()
{
//    maxNum = ui->maxNum->value();
    maxNum = ui->maxNum->value();
}
