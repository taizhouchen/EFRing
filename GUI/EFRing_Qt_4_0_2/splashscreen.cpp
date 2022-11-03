#include "splashscreen.h"
#include "ui_splash_screen.h"

SplashScreen::SplashScreen(QMainWindow *parent) : QMainWindow(parent), ui(new Ui::splashScreen)
{
    ui->setupUi(this);

    QGraphicsDropShadowEffect *shadow = new QGraphicsDropShadowEffect(this);
    shadow->setBlurRadius(20);
    shadow->setXOffset(0);
    shadow->setYOffset(0);
    shadow->setColor(QColor(0, 0, 0, 120));
    ui->circularBg->setGraphicsEffect(shadow);

    this->setWindowFlags(Qt::FramelessWindowHint);
    this->setAttribute(Qt::WA_TranslucentBackground);

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &SplashScreen::progress);
}


void SplashScreen::timerStart(double count_from_)
{
    // in sec

    if(count_from_ == 0.0) return;

    setProgressBarValue(0);
    _counter = 0;
    _speed = 100 / (count_from_ * 100);
    _timer->start(10);
    this->show();
}

void SplashScreen::showEvent(QShowEvent *event)
{

}


void SplashScreen::setProgressBarValue(double value_)
{
    QString styleSheet = "QFrame{border-radius: 75px;background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:{STOP_1} rgba(255, 0, 127, 0), stop:{STOP_2} rgba(85, 170, 255, 255));}";

    double progress = (100 - value_) / 100.0;

    QString stop_1, stop_2;

    stop_1 = QString::number(progress - 0.001);
    stop_2 = QString::number(progress);

    QString newStyleSheet = styleSheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2);

    ui->circularProgress->setStyleSheet(newStyleSheet);
}

void SplashScreen::progress()
{
    double value = _counter;

    if(value >= 100) value = 0.0;
    setProgressBarValue(value);


    if(_counter > 100){
        _timer->stop();
        emit counterStopSignal();
        this->close();
    }

    _counter += _speed;
}


