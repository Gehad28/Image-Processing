#include "dialognoise.h"
#include "ui_dialognoise.h"

Dialognoise::Dialognoise(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialognoise)
{
    ui->setupUi(this);
}

Dialognoise::~Dialognoise()
{
    delete ui;
}


void Dialognoise::on_mean_textChanged()
{
    mean = ui->mean->value();
}


void Dialognoise::on_std_textChanged()
{
    std = ui->std->value();
}
