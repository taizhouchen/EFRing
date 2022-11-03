#ifndef GESTICSTREAMINGTHREAD_H
#define GESTICSTREAMINGTHREAD_H

#include <QThread>
#include <gestic_api.h>
#include <qcustomplot.h>
#include "GOLBAL.h"
#include <gesticregressorsvm.h>
#include <splashscreen.h>
#include <datasmoothor.h>

class DataFrame
{
public:
    DataFrame();
    void toCSV(QString);
    void clear();

    void setSampleRate(int sample_rate){ _sample_rate = sample_rate; }
    void setLabel(LABELS label){ CURRENT_LABEL_ = label;}
    LABELS getLabel(){ return CURRENT_LABEL_; }

    void addData(gestic_signal_t, gestic_signal_t);
    void setCounter(int i){ COUNTER_ = i; }
    void setContinuousLabel(QList<double>);

    QList<double> getContinuousLabel();
    bool isEmpty();

    int getLength();

    QList<gestic_signal_t> getData_cic();
    QList<gestic_signal_t> getData_sd();

private:
    int _sample_rate;
    LABELS _label;
    GESTIC_DATA_TYPE _type;
    QList<gestic_signal_t> _data_sd, _data_cic;
    QList<double> _continuous_label;
//    int _counter;
};

class GesticRecorder : public QObject
{
    Q_OBJECT
public:
    GesticRecorder();
    virtual ~GesticRecorder() {}

    void init();

    void setLabel(LABELS);
    void setFileName(QString name){ _file_name = name; }
    QString getFileName(){ return _file_name; }
    LABELS getLabel(){ return _data_frame.getLabel(); }

    // recording length in ms
    void setRecordLength(int l) {
        _record_length = l;
    }

    void startRecord(bool);
    void record(const gestic_signal_t * , const gestic_signal_t *);
    bool isRecording(){ return _is_recording; }

    void saveToFile();
    void setContinuousLabel(QList<double>);

    DataFrame getDataFrame(){ return _data_frame; }

signals:
    void dataRecorded();
    void recordingStopped();

public slots:
    void onStopRecording();
    void onClearData();


private:
    LABELS _current_label;
    int _record_length;
    QTime _record_till;
    bool _is_recording;
    DataFrame _data_frame;
    QString _file_name;
    SplashScreen *splashScreen = new SplashScreen();
};

class GesticStreamingThread : public QThread
{
    Q_OBJECT

public:
    explicit GesticStreamingThread(QObject *parent = nullptr);
    void run();
    void startStreaming();
    void stopStreaming();
    bool isStreaming();

    void calibrateNow();
    void setDataType(GESTIC_DATA_TYPE new_type);
    void saveToFile();

    GesticRecorder * getRecorder(){ return _recorder; }
    void initRecorder();

    void enable();
    void disable();

    QList<SVMPoint> getSVMPointList();
    void freeRegressorX();
    void freeRegressorY();
    void freeDetector();

    GesticRegressorSVM* getSVMRegressorX();
    GesticRegressorSVM* getSVMRegressorY();
    GesticRegressorSVM* getSVMDetector();

private:
    bool _start;
    bool _ready_for_plot;
    GESTIC_DATA_TYPE _current_data_type = GESTIC_DATA_TYPE::SD;
    GesticRecorder* _recorder;
    bool _enabled;
    SVMPoint svm_point_x; // svm point for prediction

    GesticRegressorSVM* _regressorX = nullptr;
    GesticRegressorSVM* _regressorY = nullptr;
    GesticRegressorSVM* _detector = nullptr;

public slots:
    void readyForPlotSlot(bool status){
        _ready_for_plot = status;
    }

    void onAudioRecordingThreadStartRecording();
    void onTrainForRegression(QList<double>, svm_parameter);
    void onTrainForRegressionUsingSVMPoint(QList<SVMPoint>, svm_parameter);
    void onTrainForDetection(QList<SVMPoint>, svm_parameter);
    void onTrainForTracker2d(QList<double>, QList<double>, svm_parameter, svm_parameter);
    void onForceCalibrate();
    void onGetSVMPoint(QList<double>);


signals:
    void dataUpdated(const gestic_signal_t * new_data);
    void returnSVMPointSignal(QList<SVMPoint>);

};



#endif // GESTICSTREAMINGTHREAD_H
