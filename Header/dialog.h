#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = nullptr);
    ~Dialog();
    int getSize();
    int choise = 3;

private slots:
    void on_comboBox_kernelSize_activated();

private:
    Ui::Dialog *ui;
};

#endif // DIALOG_H
