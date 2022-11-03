#ifndef SPLASHSCREEN_H
#define SPLASHSCREEN_H

#include <QWidget>
#include <QtWidgets>

namespace Ui{
    class splashScreen;
}

class SplashScreen : public QMainWindow
{
    Q_OBJECT
public:
    explicit SplashScreen(QMainWindow *parent = nullptr);

    void timerStart(double);

protected:
    void showEvent(QShowEvent *event) Q_DECL_OVERRIDE;

private:
    void setProgressBarValue(double);

signals:
    void counterStopSignal();

private slots:
    void progress();

private:
    Ui::splashScreen *ui;
    double _counter, _speed;
    QTimer *_timer;

};

#endif // SPLASHSCREEN_H
