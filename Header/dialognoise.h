#ifndef DIALOGNOISE_H
#define DIALOGNOISE_H

#include <QDialog>

namespace Ui {
class Dialognoise;
}

class Dialognoise : public QDialog
{
    Q_OBJECT

public:
    explicit Dialognoise(QWidget *parent = nullptr);
    ~Dialognoise();
    float mean;
    float std;

private slots:
    void on_mean_textChanged();

    void on_std_textChanged();

private:
    Ui::Dialognoise *ui;
};

#endif // DIALOGNOISE_H
