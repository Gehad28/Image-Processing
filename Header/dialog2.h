#ifndef DIALOG2_H
#define DIALOG2_H

#include <QDialog>

namespace Ui {
class Dialog2;
}

class Dialog2 : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog2(QWidget *parent = nullptr);
    ~Dialog2();

    double radius;

private slots:
    void on_radiusBox_textChanged();

private:
    Ui::Dialog2 *ui;
};

#endif // DIALOG2_H
