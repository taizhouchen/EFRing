#include "mainwindow.h"
#include <QApplication>


int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    MainWindow w;
    w.setAttribute(Qt::WA_DeleteOnClose);
    w.show();
    return a.exec();

}
