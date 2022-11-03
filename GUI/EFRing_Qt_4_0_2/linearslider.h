#ifndef LINEARSLIDER_H
#define LINEARSLIDER_H

#include "gesticstreamingthread.h"
#include <QWidget>
#include <svm.h>
#include <datasmoothor.h>
#include "detectorwindow.h"

enum LinearSliderType { HORIZONTAL, VERTICAl };

namespace Ui {
class linearslider;
}

class LinearSlider : public QWidget
{
    Q_OBJECT

public:
    explicit LinearSlider(QWidget *parent = nullptr);
    ~LinearSlider();

    void setSpeed(int);

private:
    void showEvent(QShowEvent*);
    void closeEvent(QCloseEvent*);
    void keyPressEvent(QKeyEvent*);
    void resizeEvent(QResizeEvent* event);
    void stopRecording();

    void setDefaultSVMParam();
    void map(double *v, double v_max, double v_min, double target_max, double target_min);

private slots:
    void onTimerTimeOut();
    void onTrainingTimerOut();
    void onLoadModelActionTriggered();
    void onLoadDetectionModelActionTriggered();
    void onTrainFromFile();

private:
    Ui::linearslider *ui;
    int _minimum, _maximum, _single_step, _speed;
    int _slider_width, _slider_height;
    qint64 _current_pos;
    QTimer *_timer;
    qint64 _last_timestemp = 0, _last_pos = 0;

    QList<double> _slider_value;

    double _current_user_slider_value_normed = 0;
    struct svm_parameter _param;
    double _y_h_max_min[2];
    bool _is_recording;

    ExpMovAvg *_exp_mov_avg;
    MovAvg *_mov_avg;

    QLabel *_label_offring, *_label_none, *_label_onring;
    QList<double> _calibration_minimum_list;
    double _calibration_minimum;
    bool _is_calibrating;

    LinearSliderType _current_slider_type;

    DetectorLabel _detector_status;
    QTextBrowser *_textInstruction;


signals:
    void linearSliderReadySignal();
    void linearSliderStartRecordSignal(LinearSliderType);
    void linearSliderSaveFileSignal(QList<double>);
    void linearSliderWidgetCloseSignal();
    void linearSliderStartTrainingSignal(QList<double>, svm_parameter);
    void linearSliderStartTrainingSignalUsingSVMPoint(QList<SVMPoint>, svm_parameter);
    void forceCalibrateSignal();
    void stopRecordingSignal();
    void linearSliderLoadModelSignal();
    void linearSliderLoadDetectionModelSignal();

public slots:
    void onGesticDataRecorded();
    void onGesticRegressorSVMPredict(double, double*);
    void onGesticDetectorSVMPredict(double, double*);



};

#endif // LINEARSLIDER_H
