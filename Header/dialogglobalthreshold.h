#ifndef DIALOGGLOBALTHRESHOLD_H
#define DIALOGGLOBALTHRESHOLD_H

#include <QDialog>

namespace Ui {
class DialogGlobalThreshold;
}

class DialogGlobalThreshold : public QDialog
{
    Q_OBJECT

public:
    explicit DialogGlobalThreshold(QWidget *parent = nullptr);
    ~DialogGlobalThreshold();
    int thresholdVal;

private slots:
    void on_thresholdInput_textChanged();

private:
    Ui::DialogGlobalThreshold *ui;
};

#endif // DIALOGGLOBALTHRESHOLD_H
