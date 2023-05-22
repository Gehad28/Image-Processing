#ifndef DIALOG_CANNY_H
#define DIALOG_CANNY_H

#include <QDialog>

namespace Ui {
class Dialog_Canny;
}

class Dialog_Canny : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog_Canny(QWidget *parent = nullptr);
    ~Dialog_Canny();
    float lowThreshold;
    float highThershold;

private slots:
    void on_buttonBox_accepted();

private:
    Ui::Dialog_Canny *ui;
};

#endif // DIALOG_CANNY_H
