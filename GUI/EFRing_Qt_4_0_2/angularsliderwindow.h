#ifndef ANGULARSLIDERWINDOW_H
#define ANGULARSLIDERWINDOW_H

#include <QObject>
#include <QtWidgets>
#include <QDebug>
#include <svm.h>
#include <datasmoothor.h>

class AngularSlider : public QWidget
{
    Q_OBJECT

public :
    AngularSlider(QColor, QWidget *parent = 0);
    void setAngle(double);
    void setColor(QColor);
    double getAngle();

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

private :
    QColor _color;
    double _current_angle;
};

class AngularSliderWindow : public QWidget
{

    Q_OBJECT

public:
    AngularSliderWindow(QWidget *parent = 0);
    void setSpeed(double);

private:
    void setDefaultSVMParam();
    void map(double*, double, double, double, double);
    void stopRecording();

protected:
    void resizeEvent(QResizeEvent* event) Q_DECL_OVERRIDE;
    void showEvent(QShowEvent *event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;

signals:
    void angularSliderReadySignal();
    void angularSliderStartRecordSignal();
    void angularSliderSaveFileSignal(QList<double>);
    void angularSliderWidgetCloseSignal();
    void angularSliderStartTrainingSignal(QList<double>, svm_parameter);
    void forceCalibrateSignal();
    void stopRecordingSignal();

public slots:
    void onGesticDataRecorded();
    void onGesticRegressorSVMPredict(double, double*);

private slots:
    void onTimerTimeOut();
    void onTrainingTimerOut();

private:
    double _current_angle_slider, _current_angle_slider_user, _speed;
    AngularSlider *_angularSlider, *_angularSlider_user;
    QList<double> _slider_value;
    QTimer *_timer;
    struct svm_parameter _param;
    double _y_h_max_min[2];
    bool _is_recording;

    ExpMovAvg *_exp_mov_avg;
    MovAvg *_mov_avg;

};

#endif // ANGULARSLIDERWINDOW_H
