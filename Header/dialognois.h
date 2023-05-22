#ifndef DIALOGNOIS_H
#define DIALOGNOIS_H

#include <QDialog>

namespace Ui {
class Dialognois;
}

class Dialognois : public QDialog
{
    Q_OBJECT

public:
    explicit Dialognois(QWidget *parent = nullptr);
    ~Dialognois();
    float percent;

private slots:

    void on_noisepercent_textChanged();

private:
    Ui::Dialognois *ui;
};

#endif // DIALOGNOIS_H
