#ifndef DIALOG_EDGDETECTION_H
#define DIALOG_EDGDETECTION_H

#include <QDialog>

namespace Ui {
class Dialog_edgDetection;
}

class Dialog_edgDetection : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog_edgDetection(QWidget *parent = nullptr);
    ~Dialog_edgDetection();
    int sobel_prewitt_edgeOption;

private slots:
    void on_sobelPrewittOptions_activated(int index);


private:
    Ui::Dialog_edgDetection *ui;
};

#endif // DIALOG_EDGDETECTION_H
