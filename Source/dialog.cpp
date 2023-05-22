#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
    ui->comboBox_kernelSize->addItem("3 x 3");
    ui->comboBox_kernelSize->addItem("5 x 5");
    ui->comboBox_kernelSize->addItem("7 x 7");
}

Dialog::~Dialog()
{
    delete ui;
}

int Dialog::getSize(){
    return choise;
}

void Dialog::on_comboBox_kernelSize_activated()
{
    if(ui->comboBox_kernelSize->currentText() == "3 x 3")
        choise = 3;

    if(ui->comboBox_kernelSize->currentText() == "5 x 5")
        choise = 5;

    if(ui->comboBox_kernelSize->currentText() == "7 x 7")
        choise = 7;
}


