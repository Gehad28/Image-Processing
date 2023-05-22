#include "dialognois.h"
#include "ui_dialognois.h"

Dialognois::Dialognois(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialognois)
{
    ui->setupUi(this);
}

Dialognois::~Dialognois()
{
    delete ui;
}

void Dialognois::on_noisepercent_textChanged()
{
    percent = ui->noisepercent->value();
}
