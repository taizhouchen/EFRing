#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore>
#include <gesticstreamingthread.h>
#include "GOLBAL.h"
#include <audiorecordingthread.h>
#include <linearslider.h>
#include <angularsliderwindow.h>
#include <detectorwindow.h>
#include <tracker2dwindow.h>
#include <splashscreen.h>
#include <trialmodewindow.h>
#include <linearslidertrial.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
class QAudioRecorder;
QT_END_NAMESPACE

inline int cmpfunc(const void *a, const void *b){
    return ( *(double*)a - *(double*)b );
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void initPlot();
    bool _start;


private slots:
    void on_startStreamingBtn_clicked();
    void realtimeDataSlot(const gestic_signal_t * new_data);
    void onAudioRecorderStatusChanged(QMediaRecorder::Status);
    void onAudioRecorderDurationChanged(qint64);

    void on_calibrateBtn_clicked();
    void on_comboBox_currentIndexChanged(int index);
    void on_radioButton_0_toggled(bool checked);
    void on_radioButton_1_toggled(bool checked);
    void on_radioButton_2_toggled(bool checked);
    void on_radioButton_3_toggled(bool checked);
    void on_radioButton_4_toggled(bool checked);
    void on_radioButton_5_toggled(bool checked);
    void on_radioButton_6_toggled(bool checked);
    void on_radioButton_8_toggled(bool checked);
    void on_recordBtn_clicked();
    void on_saveBtn_clicked();
    void on_radioButton_7_toggled(bool checked);
    void on_UserIDlineEdit_textChanged(const QString &arg1);
    void on_comboBox_audiodevice_currentIndexChanged(const QString &arg1);
    void on_comboBox_sr_currentIndexChanged(const QString &arg1);
    void on_recordAudioCkbox_toggled(bool checked);
    void on_linearSliderBtn_clicked();

    void on_angularSliderBtn_clicked();

    void on_onOffDetectionBtn_clicked();

    void on_track2dBtn_clicked();

    void on_radioButton_modeoff_toggled(bool checked);

    void on_radioButton_modeon_toggled(bool checked);

    void on_trialmodelTriggered();



    void on_linearSliderTrialBtn_clicked();

    void on_radioButton_modenone_clicked(bool checked);

    void on_radioButton_modenone_toggled(bool checked);

public slots:
    void onLinearSliderReady();
    void onLinearSliderStartRecord(LinearSliderType);
    void onLinearSliderSaveFile(QList<double>);
    void onLinearSliderWidgetClose();

    void onAngularSliderReady();
    void onAngularSliderStartRecord();
    void onAngularSliderSaveFile(QList<double>);
    void onAngularSliderWidgetClose();

    void onTracker2dWindowReady();
    void onTracker2dStartRecord();
    void onTracker2dWindowClose();

    void onDetectorReady();
    void onDetectorStartRecord(bool);
    void onDetectorWindowClose();

    void onLinearSliderTrialWidgetClose();


private:
    Ui::MainWindow *ui;
    AudioRecordingThread *art = nullptr;
    LinearSlider *linearSilderWidget = new LinearSlider();
    AngularSliderWindow *angularSliderWindow = new AngularSliderWindow();
    DetectorWindow *detectorWindow = new DetectorWindow();
    Tracker2DWindow *tracker2dWindow = new Tracker2DWindow();
    TrialModeWindow *trialModeWindow = new TrialModeWindow();
    LinearSliderTrial *linearSliderTrialWidget = new LinearSliderTrial();


public:
    GesticStreamingThread *gst;

signals:
    void readyForPlotSignal(bool status);

private:
    void closeEvent(QCloseEvent*);

};


#endif // MAINWINDOW_H




