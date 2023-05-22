#ifndef DIALOGMINMAX_H
#define DIALOGMINMAX_H

#include <QDialog>

namespace Ui {
class DialogMinMax;
}

class DialogMinMax : public QDialog
{
    Q_OBJECT

public:
    explicit DialogMinMax(QWidget *parent = nullptr);
    ~DialogMinMax();
    int minNum;
    int maxNum;


private slots:
    void on_minNum_textChanged();

    void on_maxNum_textChanged();

private:
    Ui::DialogMinMax *ui;
};

#endif // DIALOGMINMAX_H
