#include "dialogglobalthreshold.h"
#include "ui_dialogglobalthreshold.h"

DialogGlobalThreshold::DialogGlobalThreshold(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogGlobalThreshold)
{
    ui->setupUi(this);
}

DialogGlobalThreshold::~DialogGlobalThreshold()
{
    delete ui;
}

void DialogGlobalThreshold::on_thresholdInput_textChanged()
{
    thresholdVal = ui->thresholdInput->value();
}


