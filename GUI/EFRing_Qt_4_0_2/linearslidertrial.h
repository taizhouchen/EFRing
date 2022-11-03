#ifndef LINEARSLIDERTRIAL_H
#define LINEARSLIDERTRIAL_H

#include <QObject>
#include <QWidget>
#include <QtWidgets>
#include <svm.h>
#include <datasmoothor.h>
#include "detectorwindow.h"
#include <math.h>
#include <iostream>

namespace Ui {
class linearslider;
}

struct fittslaw_problem_node
{
    int W, D, D_pixel, T, distance, last_pos, handle_width;
    double ID;
};

enum Filters {KalmanFilter, MovingAverage};

class LinearSliderTrial : public QWidget
{
    Q_OBJECT
public:
    explicit LinearSliderTrial(QWidget *parent = nullptr);
    ~LinearSliderTrial();
    void setFileName(QString);

protected:
    void showEvent(QShowEvent*);
    void closeEvent(QCloseEvent*);
    void keyPressEvent(QKeyEvent*);
    void resizeEvent(QResizeEvent*);
    void paintEvent(QPaintEvent*);

private:
    void valueToPixel(int, int*);
    void pixelToValue(int, int*);
    void map(double*, double, double, double, double);
    void buildFittsLawTrial();
    void nextTrial();
    void startTrial();
    void saveToFile(QString);

signals:
    void linearSliderReadySignal();
    void linearSliderWidgetCloseSignal();
    void forceCalibrateSignal();
    void linearSliderLoadModelSignal();
    void linearSliderLoadDetectionModelSignal();

private slots:
    void onSliderValueChanged();
    void onLoadModelActionTriggered();
    void onLoadDetectionModelActionTriggered();

public slots:
    void onGesticRegressorSVMPredict(double, double*);
    void onGesticDetectorSVMPredict(double, double*);

private:
    Ui::linearslider *ui;

    int _handle_width = 20;
    int _minimum = 0, _maximum = 100, _single_step = 1, _slider_width = 600, _slider_height = 100;

    QList<int> D = {33, 66};    // in value
    QList<int> W = {50, 60, 70};    // in pixel
    QList<fittslaw_problem_node> _fittslaw_problem_list;
    int _repeat = 10, _current_trial_index = -1;
    bool _fill, _ready_for_next_trial, _trial_started;

    QLabel *_label_offring, *_label_none, *_label_onring, *_label_indicator;
    DetectorLabel _detector_status;
    double _y_h_max_min[2];
    ExpMovAvg *_exp_mov_avg;
    MovAvg *_mov_avg;
    KF *_kf;
    QList<double> _calibration_minimum_list;
    double _calibration_minimum;
    bool _is_calibrating;
    QList<double> _prediction_buffer;
    int _prediction_buffer_size = 8;
    Filters _using_filter;

    qint64 _trial_start_time, _trial_end_time;
    int _prev_pos, _integrated_distance;
    QString _file_name = "";
    QTextBrowser *_textInstruction;
};

#endif // LINEARSLIDERTRIAL_H
