#ifndef DETECTORWINDOW_H
#define DETECTORWINDOW_H

#include <QObject>
#include <QWidget>
#include <QtWidgets>
#include "svm.h"
#include "GOLBAL.h"
#include <gesticregressorsvm.h>
#include <datasmoothor.h>
//#include <QtAlgorithms>

enum DetectorLabel { NO_CONTACT, OFF_RING, ON_RING };


class DetectorWindow : public QWidget
{
    Q_OBJECT
public:
    explicit DetectorWindow(QWidget *parent = nullptr);
    void setFileName(QString);

protected:
    void showEvent(QShowEvent *event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent *event) Q_DECL_OVERRIDE;
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;

private:
    void setDefaultSVMParam();
    void saveToFile(QString);
    double findMean(QList<double>);

signals:
    void detectorReadySignal();
    void detectorStartRecordSignal(bool);   // force calibrate or not
    void detectorWindowCloseSignal();

    void detectorGetSVMPointSignal(QList<double>);
    void detectorStartTrainingSignal(QList<SVMPoint>, svm_parameter);

    void forceCalibrateSignal();

    void detectorLoadModelSignal();

public slots:
    void onGesticDataRecorded();
    void onGesticDataStopRecording();
    void onReturnSVMPoint(QList<SVMPoint>);
    void onGesticDetectorSVMPredict(double, double*);
    void onTrainFromFile();

private slots:
    void onLoadModelActionTriggered();
    double labelToRegressorValue(double);

private:
    struct svm_parameter _param;
    double _current_training_label, _current_predict_label;
    QList<double> _label_list;
    QList<SVMPoint>_svm_point_none, _svm_point_offring, _svm_point_onring;
    bool _is_recording;
    QLabel *_label_offring, *_label_none, *_label_onring;

    QList<double> _calibration_base_list;
    double _calibration_base;
    bool _is_calibrating;

    QString _file_name;
    QTextBrowser *_textInstruction;
};

#endif // DETECTORWINDOW_H
