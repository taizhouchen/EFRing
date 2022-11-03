#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QTimer>
#include <QCloseEvent>
#include <linearslider.h>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    gst = new GesticStreamingThread(this);
    ui->UserIDlineEdit->setText("user0");
    gst->start();    //start the streaming thread

    art = new AudioRecordingThread(this);
    art->start();

    for (auto &device: art->getInputDevices()) {
        ui->comboBox_audiodevice->addItem(device, QVariant(device));
    }

    for (int sampleRate: art->getSupportedSampleRates()) {
        ui->comboBox_sr->addItem(QString::number(sampleRate), QVariant(
                sampleRate));
    }

    if(ui->comboBox_sr->count() > 0){
        ui->comboBox_sr->setCurrentIndex(5);
    }

    QButtonGroup *buttonGroup = new QButtonGroup(this);

    buttonGroup->addButton(ui->radioButton_modeoff);
    buttonGroup->addButton(ui->radioButton_modeon);
    buttonGroup->addButton(ui->radioButton_modenone);

    QMenuBar *menu_bar = new QMenuBar(this);
    QMenu *tool_menu = menu_bar->addMenu("Tools");

    QAction *trialMode_action = new QAction("Trial Mode", this);
    tool_menu->addAction(trialMode_action);

    connect(trialMode_action, &QAction::triggered, this, &MainWindow::on_trialmodelTriggered);

    // overwrite keyboard event in trial mode
    connect(trialModeWindow, &TrialModeWindow::startRecordSignal, this, &MainWindow::on_recordBtn_clicked);
    connect(trialModeWindow, &TrialModeWindow::saveSignal, this, &MainWindow::on_saveBtn_clicked);

    // recording from gestic sensor will be trigger from the audio recorder
    connect(art, &AudioRecordingThread::audioRecordingThreadStartRecording,
            gst, &GesticStreamingThread::onAudioRecordingThreadStartRecording);

    // for indicating the recorder status in the status bar
    connect(art->getRecorder(), &QMediaRecorder::statusChanged,
            this, &MainWindow::onAudioRecorderStatusChanged);
    connect(art->getRecorder(), &QMediaRecorder::durationChanged,
            this, &MainWindow::onAudioRecorderDurationChanged);

    initPlot();
}

MainWindow::~MainWindow()
{
    gst->quit();
    gst->wait();

    delete ui;
}

void MainWindow::initPlot()
{
    ui->customPlot->addGraph(); // red line
    ui->customPlot->graph(0)->setPen(QPen(Qt::red));
    ui->customPlot->addGraph(); // green line
    ui->customPlot->graph(1)->setPen(QPen(Qt::green));
    ui->customPlot->addGraph(); // blue line
    ui->customPlot->graph(2)->setPen(QPen(Qt::blue));
    ui->customPlot->addGraph(); // cyan line
    ui->customPlot->graph(3)->setPen(QPen(Qt::cyan));
    ui->customPlot->addGraph(); // magenta line
    ui->customPlot->graph(4)->setPen(QPen(Qt::magenta));

    QSharedPointer<QCPAxisTickerTime> timeTicker(new QCPAxisTickerTime);
    timeTicker->setTimeFormat("%h:%m:%s");
    ui->customPlot->xAxis->setTicker(timeTicker);
    ui->customPlot->axisRect()->setupFullAxesBox();
    ui->customPlot->yAxis->setRange(-1.2, 1.2);

    // make left and bottom axes transfer their ranges to right and top axes:
    connect(ui->customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->yAxis2, SLOT(setRange(QCPRange)));

    connect(gst, SIGNAL(dataUpdated(const gestic_signal_t * )), this, SLOT(realtimeDataSlot(const gestic_signal_t * )));
    connect(this, SIGNAL(readyForPlotSignal(bool)), gst, SLOT(readyForPlotSlot(bool)));
}

void MainWindow::realtimeDataSlot(const gestic_signal_t * new_data)
{
    static QTime time(QTime::currentTime());
    // calculate two new data points:
    double key = time.elapsed()/1000.0; // time elapsed since start of demo, in seconds
    static double lastPointKey = 0;
    if (key-lastPointKey > 0.01) // at most add point every 10 ms
    {
      // add data to lines:
      ui->customPlot->graph(0)->addData(key, new_data->channel[0]);
      ui->customPlot->graph(1)->addData(key, new_data->channel[1]);
      ui->customPlot->graph(2)->addData(key, new_data->channel[2]);
      ui->customPlot->graph(3)->addData(key, new_data->channel[3]);
      ui->customPlot->graph(4)->addData(key, new_data->channel[4]);

      // rescale value (vertical) axis to fit the current data:
      // ui->customPlot->graph(0)->rescaleValueAxis();
      // ui->customPlot->graph(1)->rescaleValueAxis(true);

      // auto scaling
      bool foundRange;
      double range_lower_list[5] = {ui->customPlot->graph(0)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(0)->keyAxis()->range()).lower,
                                    ui->customPlot->graph(1)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(1)->keyAxis()->range()).lower,
                                    ui->customPlot->graph(2)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(2)->keyAxis()->range()).lower,
                                    ui->customPlot->graph(3)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(3)->keyAxis()->range()).lower,
                                    ui->customPlot->graph(4)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(4)->keyAxis()->range()).lower};

      double range_upper_list[5] = {ui->customPlot->graph(0)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(0)->keyAxis()->range()).upper,
                                    ui->customPlot->graph(1)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(1)->keyAxis()->range()).upper,
                                    ui->customPlot->graph(2)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(2)->keyAxis()->range()).upper,
                                    ui->customPlot->graph(3)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(3)->keyAxis()->range()).upper,
                                    ui->customPlot->graph(4)->getValueRange(foundRange, QCP::sdBoth, ui->customPlot->graph(4)->keyAxis()->range()).upper};

      qsort(range_upper_list, 5, sizeof(double), cmpfunc);
      qsort(range_lower_list, 5, sizeof(double), cmpfunc);

      ui->customPlot->yAxis->setRange(range_upper_list[4] + 50, range_lower_list[0] - 50);
      lastPointKey = key;
    }

    if (key > 8){
        // remove data points
        ui->customPlot->graph(0)->data()->removeBefore(key-8);
        ui->customPlot->graph(1)->data()->removeBefore(key-8);
        ui->customPlot->graph(2)->data()->removeBefore(key-8);
        ui->customPlot->graph(3)->data()->removeBefore(key-8);
        ui->customPlot->graph(4)->data()->removeBefore(key-8);
    }

    // make key axis range scroll with the data (at a constant range size of 8):
    ui->customPlot->xAxis->setRange(key, 8, Qt::AlignRight);
    ui->customPlot->replot();

    emit readyForPlotSignal(true);
}


void MainWindow::on_startStreamingBtn_clicked()
{
    if(!gst->isStreaming()){
        gst->startStreaming();
        ui->startStreamingBtn->setText("Stop Streaming");
        emit readyForPlotSignal(true);
    }else{
        gst->stopStreaming();
        ui->startStreamingBtn->setText("Start Streaming");
    }
}


void MainWindow::on_calibrateBtn_clicked()
{
    gst->calibrateNow();
}


void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    switch (index) {
    case 0:
        gst->setDataType(GESTIC_DATA_TYPE::SD);
        break;
    case 1:
        gst->setDataType(GESTIC_DATA_TYPE::CIC);
        break;
    }
}


void MainWindow::on_radioButton_0_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::TAP);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::TAP);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::TAP);
        }
    }
}


void MainWindow::on_radioButton_1_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::DOUBLE_TAP);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::DOUBLE_TAP);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::DOUBLE_TAP);
        }
    }
}


void MainWindow::on_radioButton_2_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::SWIPE_LEFT);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::SWIPE_LEFT);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::SWIPE_LEFT);
        }
    }
}


void MainWindow::on_radioButton_3_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::SWIPE_RIGHT);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::SWIPE_RIGHT);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::SWIPE_RIGHT);
        }
    }
}


void MainWindow::on_radioButton_4_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::SWIPE_UP);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::SWIPE_UP);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::SWIPE_UP);
        }
    }
}


void MainWindow::on_radioButton_5_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::SWIPE_DOWN);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::SWIPE_DOWN);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::SWIPE_DOWN);
        }
    }
}


void MainWindow::on_radioButton_6_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::CHECK);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::CHECK);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::CHECK);
        }
    }
}

void MainWindow::on_radioButton_7_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::CIRCLE_CLOCKWISE);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::CIRCLE_CLOCKWISE);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::CIRCLE_CLOCKWISE);
        }
    }
}


void MainWindow::on_radioButton_8_toggled(bool checked)
{
    if(checked){
        gst->getRecorder()->setLabel(LABELS::CIRCLE_COUNTERCLOCKWISE);

        if(trialModeWindow->isVisible()){
            if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, LABELS::CIRCLE_COUNTERCLOCKWISE);
            if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, LABELS::CIRCLE_COUNTERCLOCKWISE);
        }
    }
}



void MainWindow::on_radioButton_modeoff_toggled(bool checked)
{
    gst->getRecorder()->getDataFrame().setCounter(0);

    if(trialModeWindow->isVisible()){
        if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, CURRENT_LABEL_);
        if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, CURRENT_LABEL_);
    }

    ui->radioButton_0->setEnabled(true);
    ui->radioButton_1->setEnabled(true);
    ui->radioButton_2->setEnabled(true);
    ui->radioButton_3->setEnabled(true);
    ui->radioButton_4->setEnabled(true);
    ui->radioButton_5->setEnabled(true);
    ui->radioButton_6->setEnabled(true);
    ui->radioButton_7->setEnabled(true);
    ui->radioButton_8->setEnabled(true);
}


void MainWindow::on_radioButton_modeon_toggled(bool checked)
{
    gst->getRecorder()->getDataFrame().setCounter(0);

    if(trialModeWindow->isVisible()){
        if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, CURRENT_LABEL_);
        if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, CURRENT_LABEL_);
    }

    ui->radioButton_0->setEnabled(true);
    ui->radioButton_1->setEnabled(true);
    ui->radioButton_2->setEnabled(true);
    ui->radioButton_3->setEnabled(true);
    ui->radioButton_4->setEnabled(true);
    ui->radioButton_5->setEnabled(true);
    ui->radioButton_6->setEnabled(true);
    ui->radioButton_7->setEnabled(false);
    ui->radioButton_8->setEnabled(false);

    if(ui->radioButton_7->isChecked() || ui->radioButton_8->isChecked())
        ui->radioButton_0->setChecked(true);

}

void MainWindow::on_radioButton_modenone_toggled(bool checked)
{
    gst->getRecorder()->getDataFrame().setCounter(0);
    gst->getRecorder()->setLabel(LABELS::NONE);

    ui->radioButton_0->setEnabled(false);
    ui->radioButton_1->setEnabled(false);
    ui->radioButton_2->setEnabled(false);
    ui->radioButton_3->setEnabled(false);
    ui->radioButton_4->setEnabled(false);
    ui->radioButton_5->setEnabled(false);
    ui->radioButton_6->setEnabled(false);
    ui->radioButton_7->setEnabled(false);
    ui->radioButton_8->setEnabled(false);
}

void MainWindow::on_radioButton_modenone_clicked(bool checked)
{


}


void MainWindow::on_trialmodelTriggered()
{
    if(ui->radioButton_modeon->isChecked()) trialModeWindow->setLabel(1, CURRENT_LABEL_);
    if(ui->radioButton_modeoff->isChecked()) trialModeWindow->setLabel(0, CURRENT_LABEL_);
    trialModeWindow->show();
}



void MainWindow::on_recordBtn_clicked()
{

    if(gst->getRecorder()->getLabel() == LABELS::UNDEFINE){
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invalid label");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
    else if(gst->isStreaming()){
        if(!gst->getRecorder()->isRecording()){

            TIME_STEMP_FOR_RECORD_ = QDateTime::currentDateTime();

            if(ui->recordCalibCkbox->isChecked()){
                gst->calibrateNow();
            }

            QString mode_str = "on_";

            if(ui->radioButton_modeon->isChecked()) mode_str = "on_";
            if(ui->radioButton_modeoff->isChecked()) mode_str = "off_";
            if(ui->radioButton_modenone->isChecked()) mode_str = "none_";

            if(ui->radioButton_modenone->isChecked())
                gst->getRecorder()->setRecordLength(DETECTOR_RECORD_LENGTH_);
            else
                gst->getRecorder()->setRecordLength(MICROGESTURE_LENGTH_);

            gst->getRecorder()->setFileName(DATA_FILE_PREFIXES + mode_str + ui->UserIDlineEdit->text());

            if(ui->recordAudioCkbox->isChecked()){
                art->setRecordLength(MICROGESTURE_LENGTH_);
                art->setFilePath(DATA_FILE_PREFIXES + mode_str + ui->UserIDlineEdit->text());

                // gestic record will be triggered by the audio recorder
                gst->getRecorder()->onClearData();
                art->startRecord();
            }else{
                gst->getRecorder()->onClearData();
                gst->getRecorder()->startRecord(true);
            }

        }
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Please start streaming before recording.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}


void MainWindow::on_saveBtn_clicked()
{
    if(ui->recordAudioCkbox->isChecked()){
        if(art->keepTheFile()){
            gst->getRecorder()->saveToFile();
        }
    }else{
        gst->getRecorder()->saveToFile();
    }
}


void MainWindow::on_UserIDlineEdit_textChanged(const QString &arg1)
{
    gst->getRecorder()->getDataFrame().setCounter(0);
}


void MainWindow::on_comboBox_audiodevice_currentIndexChanged(const QString &arg1)
{
    art->setInputDevice(arg1);
}


void MainWindow::on_comboBox_sr_currentIndexChanged(const QString &arg1)
{
    art->setSampleRate(arg1.toInt());
}


void MainWindow::on_recordAudioCkbox_toggled(bool checked)
{
    if(checked){
        connect(art->getRecorder(), &QMediaRecorder::statusChanged, this, &MainWindow::onAudioRecorderStatusChanged);
        connect(art->getRecorder(), &QMediaRecorder::durationChanged, this, &MainWindow::onAudioRecorderDurationChanged);

        ui->comboBox_audiodevice->setDisabled(false);
        ui->comboBox_sr->setDisabled(false);
    }else{
        disconnect(art->getRecorder(), &QMediaRecorder::statusChanged, this, &MainWindow::onAudioRecorderStatusChanged);
        disconnect(art->getRecorder(), &QMediaRecorder::durationChanged, this, &MainWindow::onAudioRecorderDurationChanged);

        ui->comboBox_audiodevice->setDisabled(true);
        ui->comboBox_sr->setDisabled(true);
    }
}

void MainWindow::onLinearSliderReady()
{
    qDebug() << "on linear slider ready";
    if(gst->isStreaming()){
        gst->getRecorder()->init();
    }
}

void MainWindow::onLinearSliderStartRecord(LinearSliderType type_)
{
    qDebug() << "on linear slider start";
    if(gst->isStreaming() && !gst->getRecorder()->isRecording()){

        TIME_STEMP_FOR_RECORD_ = QDateTime::currentDateTime();

        if(ui->recordCalibCkbox->isChecked()){
            gst->calibrateNow();
        }

        CURRENT_LABEL_ = LABELS::CONTINUOUS_LINEAR_SLIDER;

        gst->getRecorder()->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);

        QString file_name_suffix = "_linearSlider";

        if(type_ == LinearSliderType::HORIZONTAL)
            file_name_suffix = "_horizontalLinearSlider";
        if(type_ == LinearSliderType::VERTICAl)
            file_name_suffix = "_verticalLinearSlider";

        gst->getRecorder()->setFileName(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + file_name_suffix);

        if(ui->recordAudioCkbox->isChecked()){
            art->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);
            art->setFilePath(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + file_name_suffix);

            // gestic record will be triggered by the audio recorder
            art->startRecord();
        }else{
            gst->getRecorder()->startRecord(false);
        }
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Please start streaming before recording.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}

void MainWindow::onLinearSliderSaveFile(QList<double> slider_value)
{
    if(gst->getRecorder()->isRecording()) return;

    qDebug() << "on linear slider save file";

    gst->getRecorder()->setContinuousLabel(slider_value);
    gst->getRecorder()->saveToFile();
}


void MainWindow::onAudioRecorderStatusChanged(QMediaRecorder::Status new_status)
{
    QString statusMessage;

    switch (new_status) {
    case QMediaRecorder::RecordingStatus:
        statusMessage = tr("Recording");
        break;
    case QMediaRecorder::PausedStatus:
        statusMessage = tr("Paused");
        break;
    case QMediaRecorder::UnloadedStatus:
        statusMessage = tr("Unloaded");
        break;
    case QMediaRecorder::LoadedStatus:
        statusMessage = tr("Stopped");
    default:
        break;
    }

    if (art->getRecorder()->error() == QMediaRecorder::NoError)
        ui->statusbar->showMessage(statusMessage);
}

void MainWindow::onAudioRecorderDurationChanged(qint64 duration)
{
    if (art->getRecorder()->error() != QMediaRecorder::NoError)
        return;

    ui->statusbar->showMessage(tr("Recorded %1 ms").arg(duration));
}


void MainWindow::on_linearSliderBtn_clicked()
{
    connect(linearSilderWidget, &LinearSlider::linearSliderWidgetCloseSignal,
            this, &MainWindow::onLinearSliderWidgetClose);
    connect(linearSilderWidget, &LinearSlider::linearSliderReadySignal,
            this, &MainWindow::onLinearSliderReady);
    connect(linearSilderWidget, &LinearSlider::linearSliderStartRecordSignal,
            this, &MainWindow::onLinearSliderStartRecord);
    connect(linearSilderWidget, &LinearSlider::linearSliderSaveFileSignal,
            this, &MainWindow::onLinearSliderSaveFile);

    connect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            linearSilderWidget, &LinearSlider::onGesticDataRecorded);
    connect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSilderWidget, &LinearSlider::onGesticRegressorSVMPredict);

    connect(linearSilderWidget, &LinearSlider::linearSliderStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForRegression);
    connect(linearSilderWidget, &LinearSlider::linearSliderStartTrainingSignalUsingSVMPoint,
            gst, &GesticStreamingThread::onTrainForRegressionUsingSVMPoint);
    connect(linearSilderWidget, &LinearSlider::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    connect(linearSilderWidget, &LinearSlider::stopRecordingSignal,
            gst->getRecorder(), &GesticRecorder::onStopRecording);

    connect(linearSilderWidget, &LinearSlider::linearSliderLoadModelSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);

    connect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSilderWidget, &LinearSlider::onGesticDetectorSVMPredict);
    connect(linearSilderWidget, &LinearSlider::linearSliderLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

    linearSilderWidget->show();
}

void MainWindow::onLinearSliderWidgetClose()
{
    gst->freeRegressorX();
    gst->freeDetector();

    disconnect(linearSilderWidget, &LinearSlider::linearSliderWidgetCloseSignal,
               this, &MainWindow::onLinearSliderWidgetClose);
    disconnect(linearSilderWidget, &LinearSlider::linearSliderReadySignal,
            this, &MainWindow::onLinearSliderReady);
    disconnect(linearSilderWidget, &LinearSlider::linearSliderStartRecordSignal,
            this, &MainWindow::onLinearSliderStartRecord);
    disconnect(linearSilderWidget, &LinearSlider::linearSliderSaveFileSignal,
            this, &MainWindow::onLinearSliderSaveFile);

    disconnect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            linearSilderWidget, &LinearSlider::onGesticDataRecorded);
    disconnect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSilderWidget, &LinearSlider::onGesticRegressorSVMPredict);

    disconnect(linearSilderWidget, &LinearSlider::linearSliderStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForRegression);
    disconnect(linearSilderWidget, &LinearSlider::linearSliderStartTrainingSignalUsingSVMPoint,
            gst, &GesticStreamingThread::onTrainForRegressionUsingSVMPoint);
    disconnect(linearSilderWidget, &LinearSlider::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    disconnect(linearSilderWidget, &LinearSlider::stopRecordingSignal,
            gst->getRecorder(), &GesticRecorder::onStopRecording);

    disconnect(linearSilderWidget, &LinearSlider::linearSliderLoadModelSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);

    disconnect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSilderWidget, &LinearSlider::onGesticDetectorSVMPredict);
    disconnect(linearSilderWidget, &LinearSlider::linearSliderLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

}

void MainWindow::on_angularSliderBtn_clicked()
{
    connect(angularSliderWindow, &AngularSliderWindow::angularSliderWidgetCloseSignal,
            this, &MainWindow::onAngularSliderWidgetClose);
    connect(angularSliderWindow, &AngularSliderWindow::angularSliderReadySignal,
            this, &MainWindow::onAngularSliderReady);
    connect(angularSliderWindow, &AngularSliderWindow::angularSliderStartRecordSignal,
           this, &MainWindow::onAngularSliderStartRecord);
    connect(angularSliderWindow, &AngularSliderWindow::angularSliderSaveFileSignal,
            this, &MainWindow::onAngularSliderSaveFile);

    connect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            angularSliderWindow, &AngularSliderWindow::onGesticDataRecorded);
    connect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            angularSliderWindow, &AngularSliderWindow::onGesticRegressorSVMPredict);

    connect(angularSliderWindow, &AngularSliderWindow::angularSliderStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForRegression);
    connect(angularSliderWindow, &AngularSliderWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    connect(angularSliderWindow, &AngularSliderWindow::stopRecordingSignal,
            gst->getRecorder(), &GesticRecorder::onStopRecording);

    angularSliderWindow->show();
}

void MainWindow::onAngularSliderWidgetClose()
{
    gst->freeRegressorX();

    disconnect(angularSliderWindow, &AngularSliderWindow::angularSliderWidgetCloseSignal,
            this, &MainWindow::onAngularSliderWidgetClose);
    disconnect(angularSliderWindow, &AngularSliderWindow::angularSliderReadySignal,
            this, &MainWindow::onAngularSliderReady);
    disconnect(angularSliderWindow, &AngularSliderWindow::angularSliderStartRecordSignal,
           this, &MainWindow::onAngularSliderStartRecord);
    disconnect(angularSliderWindow, &AngularSliderWindow::angularSliderSaveFileSignal,
            this, &MainWindow::onAngularSliderSaveFile);

    disconnect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            angularSliderWindow, &AngularSliderWindow::onGesticDataRecorded);
    disconnect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            angularSliderWindow, &AngularSliderWindow::onGesticRegressorSVMPredict);

    disconnect(angularSliderWindow, &AngularSliderWindow::angularSliderStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForRegression);
    disconnect(angularSliderWindow, &AngularSliderWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    disconnect(angularSliderWindow, &AngularSliderWindow::stopRecordingSignal,
               gst->getRecorder(), &GesticRecorder::onStopRecording);
}

void MainWindow::onTracker2dWindowReady()
{
    qDebug() << "on track 2d window ready";
    if(gst->isStreaming()){
        gst->getRecorder()->init();
    }
}

void MainWindow::onTracker2dStartRecord()
{
    qDebug() << "on tracker 2d start record";
    if(gst->isStreaming() && !gst->getRecorder()->isRecording()){

        TIME_STEMP_FOR_RECORD_ = QDateTime::currentDateTime();

        if(ui->recordCalibCkbox->isChecked()){
            gst->calibrateNow();
        }

        CURRENT_LABEL_ = LABELS::CONTINUOUS_LINEAR_SLIDER;

        gst->getRecorder()->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);
        gst->getRecorder()->setFileName(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_tracker2d");

        if(ui->recordAudioCkbox->isChecked()){
            art->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);
            art->setFilePath(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_tracker2d");

            // gestic record will be triggered by the audio recorder
            art->startRecord();
        }else{
            gst->getRecorder()->startRecord(false);
        }
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Please start streaming before recording.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}


void MainWindow::on_onOffDetectionBtn_clicked()
{
    connect(detectorWindow, &DetectorWindow::detectorReadySignal,
            this, &MainWindow::onDetectorReady);
    connect(detectorWindow, &DetectorWindow::detectorStartRecordSignal,
            this, &MainWindow::onDetectorStartRecord);
    connect(detectorWindow, &DetectorWindow::detectorWindowCloseSignal,
            this, &MainWindow::onDetectorWindowClose);

    connect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            detectorWindow, &DetectorWindow::onGesticDataRecorded);

    connect(gst->getRecorder(), &GesticRecorder::recordingStopped,
            detectorWindow, &DetectorWindow::onGesticDataStopRecording);
    connect(detectorWindow, &DetectorWindow::detectorGetSVMPointSignal,
            gst, &GesticStreamingThread::onGetSVMPoint);
    connect(gst, &GesticStreamingThread::returnSVMPointSignal,
            detectorWindow, &DetectorWindow::onReturnSVMPoint);

    connect(detectorWindow, &DetectorWindow::detectorStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForDetection);
    connect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
             detectorWindow, &DetectorWindow::onGesticDetectorSVMPredict);
    connect(detectorWindow, &DetectorWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);

    connect(detectorWindow, &DetectorWindow::detectorLoadModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

    detectorWindow->setFileName(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_detector");
    detectorWindow->show();
}

void MainWindow::onDetectorWindowClose()
{
    gst->freeDetector();

    disconnect(detectorWindow, &DetectorWindow::detectorReadySignal,
            this, &MainWindow::onDetectorReady);
    disconnect(detectorWindow, &DetectorWindow::detectorStartRecordSignal,
            this, &MainWindow::onDetectorStartRecord);
    disconnect(detectorWindow, &DetectorWindow::detectorWindowCloseSignal,
            this, &MainWindow::onDetectorWindowClose);

    disconnect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            detectorWindow, &DetectorWindow::onGesticDataRecorded);

    disconnect(gst->getRecorder(), &GesticRecorder::recordingStopped,
            detectorWindow, &DetectorWindow::onGesticDataStopRecording);
    disconnect(detectorWindow, &DetectorWindow::detectorGetSVMPointSignal,
            gst, &GesticStreamingThread::onGetSVMPoint);
    disconnect(gst, &GesticStreamingThread::returnSVMPointSignal,
            detectorWindow, &DetectorWindow::onReturnSVMPoint);

    disconnect(detectorWindow, &DetectorWindow::detectorStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForDetection);
    disconnect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
             detectorWindow, &DetectorWindow::onGesticDetectorSVMPredict);
    disconnect(detectorWindow, &DetectorWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);

    disconnect(detectorWindow, &DetectorWindow::detectorLoadModelSignal,
               gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);
}


void MainWindow::on_track2dBtn_clicked()
{
    connect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowCloseSignal,
            this, &MainWindow::onTracker2dWindowClose);
    connect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowReadySignal,
            this, &MainWindow::onTracker2dWindowReady);
    connect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowStartRecordSignal,
            this, &MainWindow::onTracker2dStartRecord);
//    connect(linearSilderWidget, &LinearSlider::linearSliderSaveFileSignal,
//            this, &MainWindow::onLinearSliderSaveFile);

    connect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            tracker2dWindow, &Tracker2DWindow::onGesticDataRecorded);
    connect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            tracker2dWindow, &Tracker2DWindow::onGesticRegressorSVMPredictX);
    connect(gst->getSVMRegressorY(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            tracker2dWindow, &Tracker2DWindow::onGesticRegressorSVMPredictY);

    connect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForTracker2d);
    connect(tracker2dWindow, &Tracker2DWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    connect(tracker2dWindow, &Tracker2DWindow::stopRecordingSignal,
            gst->getRecorder(), &GesticRecorder::onStopRecording);

    connect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadModelXSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);
    connect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadModelYSignal,
            gst->getSVMRegressorY(), &GesticRegressorSVM::onLoadModelFromFile);

//    connect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
//            linearSilderWidget, &LinearSlider::onGesticDetectorSVMPredict);
    connect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

    tracker2dWindow->show();
}

void MainWindow::onTracker2dWindowClose()
{
    gst->freeRegressorX();
    gst->freeRegressorY();

    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowCloseSignal,
            this, &MainWindow::onTracker2dWindowClose);
    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowReadySignal,
            this, &MainWindow::onTracker2dWindowReady);
    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowStartRecordSignal,
            this, &MainWindow::onTracker2dStartRecord);
//    disconnect(linearSilderWidget, &LinearSlider::linearSliderSaveFileSignal,
//            this, &MainWindow::onLinearSliderSaveFile);

    disconnect(gst->getRecorder(), &GesticRecorder::dataRecorded,
            tracker2dWindow, &Tracker2DWindow::onGesticDataRecorded);
    disconnect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            tracker2dWindow, &Tracker2DWindow::onGesticRegressorSVMPredictX);
    disconnect(gst->getSVMRegressorY(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            tracker2dWindow, &Tracker2DWindow::onGesticRegressorSVMPredictY);

    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dWindowStartTrainingSignal,
            gst, &GesticStreamingThread::onTrainForTracker2d);
    disconnect(tracker2dWindow, &Tracker2DWindow::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);
    disconnect(tracker2dWindow, &Tracker2DWindow::stopRecordingSignal,
            gst->getRecorder(), &GesticRecorder::onStopRecording);

    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadModelXSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);
    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadModelYSignal,
            gst->getSVMRegressorY(), &GesticRegressorSVM::onLoadModelFromFile);

//    disconnect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
//            linearSilderWidget, &LinearSlider::onGesticDetectorSVMPredict);
    disconnect(tracker2dWindow, &Tracker2DWindow::tracker2dLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

}

void MainWindow::onDetectorReady()
{
    qDebug() << "on detector ready";
    if(gst->isStreaming()){
        gst->getRecorder()->init();
    }
}

void MainWindow::onDetectorStartRecord(bool calibrate_)
{
    qDebug() << "on detector start record";
    if(gst->isStreaming() && !gst->getRecorder()->isRecording()){

        TIME_STEMP_FOR_RECORD_ = QDateTime::currentDateTime();

        if(calibrate_){
            gst->calibrateNow();
        }

        CURRENT_LABEL_ = LABELS::CONTINUOUS_LINEAR_SLIDER;

        gst->getRecorder()->setRecordLength(DETECTOR_RECORD_LENGTH_);
        gst->getRecorder()->setFileName(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_detector");

        if(ui->recordAudioCkbox->isChecked()){
            art->setRecordLength(DETECTOR_RECORD_LENGTH_);
            art->setFilePath(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_detector");

            // gestic record will be triggered by the audio recorder
            gst->getRecorder()->onClearData();
            art->startRecord();
        }else{
            gst->getRecorder()->onClearData();
            gst->getRecorder()->startRecord(true);
        }
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Please start streaming before recording.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}



void MainWindow::onAngularSliderReady()
{
    qDebug() << "on angular slider ready";
    if(gst->isStreaming()){
        gst->getRecorder()->init();
    }
}

void MainWindow::onAngularSliderStartRecord()
{
    qDebug() << "on angular slider start";
    if(gst->isStreaming() && !gst->getRecorder()->isRecording()){

        TIME_STEMP_FOR_RECORD_ = QDateTime::currentDateTime();

        if(ui->recordCalibCkbox->isChecked()){
            gst->calibrateNow();
        }

        CURRENT_LABEL_ = LABELS::CONTINUOUS_LINEAR_SLIDER;

        gst->getRecorder()->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);
        gst->getRecorder()->setFileName(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_angularSlider");

        if(ui->recordAudioCkbox->isChecked()){
            art->setRecordLength(LINEARGESTURE_LENGTH_MAXIMUM_);
            art->setFilePath(DATA_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_angularSlider");

            // gestic record will be triggered by the audio recorder
            art->startRecord();
        }else{
            gst->getRecorder()->startRecord(false);
        }
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Please start streaming before recording.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}

void MainWindow::onAngularSliderSaveFile(QList<double> slider_value)
{
    if(gst->getRecorder()->isRecording()) return;

    qDebug() << "on angular slider save file";

    gst->getRecorder()->setContinuousLabel(slider_value);
    gst->getRecorder()->saveToFile();
}



void MainWindow::on_linearSliderTrialBtn_clicked()
{
    connect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderWidgetCloseSignal,
            this, &MainWindow::onLinearSliderTrialWidgetClose);
    connect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderReadySignal,
            this, &MainWindow::onLinearSliderReady);
    connect(linearSliderTrialWidget, &LinearSliderTrial::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);

    connect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSliderTrialWidget, &LinearSliderTrial::onGesticRegressorSVMPredict);
    connect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSliderTrialWidget, &LinearSliderTrial::onGesticDetectorSVMPredict);

    connect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderLoadModelSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);
    connect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);

    linearSliderTrialWidget->setFileName(RESULT_FILE_PREFIXES + ui->UserIDlineEdit->text() + "_linearSliderTrial");
    linearSliderTrialWidget->show();
}

void MainWindow::onLinearSliderTrialWidgetClose()
{
    gst->freeRegressorX();
    gst->freeDetector();

    disconnect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderWidgetCloseSignal,
            this, &MainWindow::onLinearSliderTrialWidgetClose);
    disconnect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderReadySignal,
            this, &MainWindow::onLinearSliderReady);
    disconnect(linearSliderTrialWidget, &LinearSliderTrial::forceCalibrateSignal,
            gst, &GesticStreamingThread::onForceCalibrate);

    disconnect(gst->getSVMRegressorX(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSliderTrialWidget, &LinearSliderTrial::onGesticRegressorSVMPredict);
    disconnect(gst->getSVMDetector(), &GesticRegressorSVM::gesticRegressorSVMPredict,
            linearSliderTrialWidget, &LinearSliderTrial::onGesticDetectorSVMPredict);

    disconnect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderLoadModelSignal,
            gst->getSVMRegressorX(), &GesticRegressorSVM::onLoadModelFromFile);
    disconnect(linearSliderTrialWidget, &LinearSliderTrial::linearSliderLoadDetectionModelSignal,
            gst->getSVMDetector(), &GesticRegressorSVM::onLoadModelFromFile);
}



void MainWindow::closeEvent(QCloseEvent* e){
    qDebug() << "close";

    if(gst->isRunning())
    {
        gst->disable();
        gst->exit(0);
        connect(gst, SIGNAL(finished()), this, SLOT(close()), Qt::UniqueConnection);
    }

    if(art->isRunning())
    {
        art->disable();
        gst->exit(0);
        connect(art, SIGNAL(finished()), this, SLOT(close()), Qt::UniqueConnection);
    }

    if(gst->isRunning() || art->isRunning()){
        e->ignore();
    }
}







