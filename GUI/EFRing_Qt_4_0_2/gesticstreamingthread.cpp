#include "gesticstreamingthread.h"
#include <QDebug>
#include <mygestic.h>
#include <QMessageBox>

GesticStreamingThread::GesticStreamingThread(QObject *parent) : QThread(parent)
{
    enable();
    _start = false;
    _recorder = new GesticRecorder();

    _regressorY = new GesticRegressorSVM();
    _detector = new GesticRegressorSVM();
    _regressorX = new GesticRegressorSVM();

    _regressorX->enableSmoother();
    _regressorY->enableSmoother();
    _detector->disableSmoother();

    _regressorX->setSlidingWindowParam(3, 1);

    QList<SVMFeatures> regressorX_feature_list_;
    regressorX_feature_list_.append(SVMFeatures::SD_SVM);
//    regressorX_feature_list_.append(SVMFeatures::CIC_SVM);
//    regressorX_feature_list_.append(SVMFeatures::SD_DIFF_SVM);
    _regressorX->setFeatureList(regressorX_feature_list_);

    _regressorY->setSlidingWindowParam(3, 1);

    QList<SVMFeatures> regressorY_feature_list_;
//    regressorX_feature_list_.append(SVMFeatures::SD_SVM);
    regressorY_feature_list_.append(SVMFeatures::SD_DIFF_SVM);
//    regressorY_feature_list_.append(SVMFeatures::CIC_DIFF_SVM);
    _regressorY->setFeatureList(regressorY_feature_list_);

    _detector->setSlidingWindowParam(3, 1);

    QList<SVMFeatures> detector_feature_list_;
    detector_feature_list_.append(SVMFeatures::SD_DIFF_SVM);
//    detector_feature_list_.append(SVMFeatures::SD_SVM);
//    detector_feature_list_.append(SVMFeatures::CIC_DIFF_SVM);
    _detector->setFeatureList(detector_feature_list_);
}

void GesticStreamingThread::run(){
    while(_enabled){
        while(_start){
            while(!gestic_data_stream_update(gestic, 0)){

//                qDebug() << QString("%1, %2, %3, %4, %5")
//                            .arg(cic->channel[0])
//                            .arg(cic->channel[1])
//                            .arg(cic->channel[2])
//                            .arg(cic->channel[3])
//                            .arg(cic->channel[4]);


                if(_recorder->isRecording()){
                    _recorder->record(sd, cic);
                }

                if((_regressorX != nullptr && _regressorX->isTrained()) ||
                   (_regressorY != nullptr && _regressorY->isTrained()) ||
                   (_detector != nullptr && _detector->isTrained())){

                    svm_point_x.sd_c0 = sd->channel[0];
                    svm_point_x.sd_c1 = sd->channel[1];
                    svm_point_x.sd_c2 = sd->channel[2];
                    svm_point_x.sd_c3 = sd->channel[3];
                    svm_point_x.sd_c4 = sd->channel[4];

                    svm_point_x.cic_c0 = cic->channel[0];
                    svm_point_x.cic_c1 = cic->channel[1];
                    svm_point_x.cic_c2 = cic->channel[2];
                    svm_point_x.cic_c3 = cic->channel[3];
                    svm_point_x.cic_c4 = cic->channel[4];

                    if(_regressorX->isTrained())
                        _regressorX->predict(svm_point_x);
                    if(_regressorY->isTrained())
                        _regressorY->predict(svm_point_x);
                    if(_detector->isTrained())
                        _detector->predict(svm_point_x);
                }

                if(_ready_for_plot){
                    switch (_current_data_type) {
                        case GESTIC_DATA_TYPE::SD:
                            emit dataUpdated(sd);
                            break;
                        case GESTIC_DATA_TYPE::CIC:
                            emit dataUpdated(cic);
                            break;
                        default:
                            emit dataUpdated(sd);
                            break;
                    }
                    _ready_for_plot = false;
                }
            }
            msleep(5);
        }
    }
}

void GesticStreamingThread::startStreaming()
{
    gestic_init();

    QString message_str = "";
    switch (my_gestic_error) {
        case MY_GESTIC_NOT_DEVICE:
            message_str = "Could not open connection to device.";
            break;
        case MY_GESTIC_NOT_RESET_TO_DEFAULT:
            message_str = "Could not reset device to default state.";
            break;
        case MY_GESTIC_NOT_OUTPUT_MASKING:
            message_str = "Could not set output-mask for streaming.";
            break;
        case MY_GESTIC_NO_ERROR:
            _start = true;
            qDebug() << "Streaming started";
            return;
            break;
    }

    QMessageBox msgBox;
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText(message_str);
    msgBox.setWindowTitle("ERROR");
    msgBox.exec();
}

void GesticStreamingThread::stopStreaming()
{
    gestic_release();
    _start = false;
    qDebug() << "Streaming stopped";
}

bool GesticStreamingThread::isStreaming()
{
    return _start;
}

void GesticStreamingThread::calibrateNow()
{
    if(GESTIC_INITIALIZED){
        gestic_force_calibration(gestic, 100);
        qDebug() << "Calibrate sucessfully";
    }
}

void GesticStreamingThread::setDataType(GESTIC_DATA_TYPE new_type)
{
    _current_data_type = new_type;
}

void GesticStreamingThread::saveToFile()
{
    _recorder->saveToFile();
}

void GesticStreamingThread::initRecorder()
{
    _recorder->init();
}

void GesticStreamingThread::enable()
{
    _enabled = true;
}

void GesticStreamingThread::disable()
{
    _enabled = false;
}

QList<SVMPoint> GesticStreamingThread::getSVMPointList()
{
    // convert from DataFrame to SVMPoint
    QList<SVMPoint> svm_point_list;
    if(getRecorder()->getDataFrame().getContinuousLabel().size() == 0){

    }else{
        QList<gestic_signal_t> data_cic = getRecorder()->getDataFrame().getData_cic();
        QList<gestic_signal_t> data_sd = getRecorder()->getDataFrame().getData_sd();
        QList<double> continuous_label = getRecorder()->getDataFrame().getContinuousLabel();
        int length = getRecorder()->getDataFrame().getLength();

        for(int i = 0; i < length; i++){
            SVMPoint p;
            p.sd_c0 = data_sd[i].channel[0];
            p.sd_c1 = data_sd[i].channel[1];
            p.sd_c2 = data_sd[i].channel[2];
            p.sd_c3 = data_sd[i].channel[3];
            p.sd_c4 = data_sd[i].channel[4];
            p.cic_c0 = data_cic[i].channel[0];
            p.cic_c1 = data_cic[i].channel[1];
            p.cic_c2 = data_cic[i].channel[2];
            p.cic_c3 = data_cic[i].channel[3];
            p.cic_c4 = data_cic[i].channel[4];
            p.y = continuous_label[i];
            svm_point_list.push_back(p);
        }
    }
    return svm_point_list;
}

void GesticStreamingThread::freeRegressorX()
{
    qDebug() << (_regressorX == nullptr);
    if(_regressorX != nullptr)
        _regressorX->release();
}

void GesticStreamingThread::freeRegressorY()
{
    qDebug() << (_regressorY == nullptr);
    if(_regressorY != nullptr)
        _regressorY->release();
}

void GesticStreamingThread::freeDetector()
{
    qDebug() << (_detector == nullptr);
    if(_detector != nullptr)
        _detector->release();
}

GesticRegressorSVM *GesticStreamingThread::getSVMRegressorY()
{
    return _regressorY;
}

GesticRegressorSVM *GesticStreamingThread::getSVMRegressorX()
{
    return _regressorX;
}

GesticRegressorSVM *GesticStreamingThread::getSVMDetector()
{
    return _detector;
}

void GesticStreamingThread::onAudioRecordingThreadStartRecording()
{
    _recorder->startRecord(true);
}

void GesticStreamingThread::onTrainForRegression(QList<double> slider_value, svm_parameter param)
{
    if(getRecorder()->isRecording()) return;
    qDebug() << "On start training for regression" ;

    getRecorder()->setContinuousLabel(slider_value);
    QList<SVMPoint> svm_point_list = getSVMPointList();
    if(svm_point_list.size() == 0) {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invaid data for regression.");
        msgBox.setWindowTitle("onTrainForRegression Message");
        msgBox.exec();
        return;
    }

    _regressorX->setParam(param);
    _regressorX->setData(svm_point_list);
    _regressorX->setFileName(getRecorder()->getFileName());
    _regressorX->buildSVMProblem();
    _regressorX->train();
}

void GesticStreamingThread::onTrainForRegressionUsingSVMPoint(QList<SVMPoint> svm_point_list_, svm_parameter param_)
{
    if(getRecorder()->isRecording()) return;
    qDebug() << "On start training for regression" ;

    if(svm_point_list_.size() == 0) {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invaid data for detection training.");
        msgBox.setWindowTitle("onTrainForDetection Message");
        msgBox.exec();
        return;
    }

    _regressorX->setParam(param_);
    _regressorX->setData(svm_point_list_);
    _regressorX->setFileName(getRecorder()->getFileName());
    _regressorX->buildSVMProblem();
    _regressorX->train();
}

void GesticStreamingThread::onForceCalibrate()
{
    calibrateNow();
}

void GesticStreamingThread::onGetSVMPoint(QList<double> label_)
{
    if(getRecorder()->isRecording()) return;

    getRecorder()->setContinuousLabel(label_);
    QList<SVMPoint> svm_point_list = getSVMPointList();

    emit returnSVMPointSignal(svm_point_list);
}

void GesticStreamingThread::onTrainForDetection(QList<SVMPoint> svm_point_list_, svm_parameter param_)
{
    if(getRecorder()->isRecording()) return;
    qDebug() << "On start training for detection" ;

    if(svm_point_list_.size() == 0) {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invaid data for detection training.");
        msgBox.setWindowTitle("onTrainForDetection Message");
        msgBox.exec();
        return;
    }

    _detector->setParam(param_);
    _detector->setData(svm_point_list_);
    _detector->setFileName(getRecorder()->getFileName());
    _detector->buildSVMProblem();
    _detector->train();
}

void GesticStreamingThread::onTrainForTracker2d(QList<double> pos_x_list_, QList<double> pos_y_list_, svm_parameter param_x_, svm_parameter param_y_)
{
    if(getRecorder()->isRecording()) return;
    qDebug() << "On start training for tracker2d" ;

    getRecorder()->setContinuousLabel(pos_x_list_);
    QList<SVMPoint> svm_point_list_x = getSVMPointList();
    if(svm_point_list_x.size() == 0) {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invaid data for regression X.");
        msgBox.setWindowTitle("onTrainForTracker2d Message");
        msgBox.exec();
        return;
    }

    _regressorX->setParam(param_x_);
    _regressorX->setData(svm_point_list_x);
    _regressorX->setFileName(getRecorder()->getFileName() + "_x");
    _regressorX->buildSVMProblem();
    _regressorX->train();


    getRecorder()->setContinuousLabel(pos_y_list_);
    QList<SVMPoint> svm_point_list_y = getSVMPointList();
    if(svm_point_list_y.size() == 0) {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Invaid data for regression Y.");
        msgBox.setWindowTitle("onTrainForTracker2d Message");
        msgBox.exec();
        return;
    }

    _regressorY->setParam(param_y_);
    _regressorY->setData(svm_point_list_y);
    _regressorY->setFileName(getRecorder()->getFileName() + "_y");
    _regressorY->buildSVMProblem();
    _regressorY->train();
}


GesticRecorder::GesticRecorder() {
//    _record_length = RECORDING_LENGTH_;
    _is_recording = false;
}

void GesticRecorder::init()
{
    _is_recording = false;
    _data_frame.setLabel(LABELS::UNDEFINE);
    _data_frame.clear();
    _file_name = "";
    _record_length = 1000;
}

void GesticRecorder::setLabel(LABELS new_label){
    _data_frame.setLabel(new_label);
}

void GesticRecorder::startRecord(bool count_down_)
{
    if(_is_recording){
        return;
    }
    _record_till = QTime::currentTime().addMSecs(_record_length);
    _is_recording = true;

    if(count_down_){

        QPoint new_pos;

        // if there is a secondary screen detected, move the countdown widget to the secondary screen
        QApplication::desktop()->screenCount() == 1 ?
            new_pos = QPoint(QApplication::desktop()->screen()->rect().center().x() - splashScreen->rect().width()/2,
                             QApplication::desktop()->screen()->rect().height()*7/8 - splashScreen->rect().height()/2) :
            new_pos = QPoint(QApplication::desktop()->availableGeometry(1).center().x() - splashScreen->rect().width()/2,
                             QApplication::desktop()->availableGeometry(1).height()*7/8 - splashScreen->rect().height()/2);

        splashScreen->move(new_pos);
        splashScreen->timerStart(_record_length/1000 - 0.1);
    }
//    _data_frame.clear();
    qDebug() << "GestIC record started!";
}

void GesticRecorder::record(const gestic_signal_t * data_sd, const gestic_signal_t * data_cic)
{
    _data_frame.addData(*data_sd, *data_cic);
    emit dataRecorded();

    if(QTime::currentTime() > _record_till){
        _is_recording = false;
        qDebug() << "GectIC record finished!";

        emit recordingStopped();
    }
}

void GesticRecorder::saveToFile()
{
    if(!_is_recording){
        if(_file_name != "") _data_frame.toCSV(_file_name);
        else qDebug() << "Filename undefined";
    }
}

void GesticRecorder::setContinuousLabel(QList<double> label)
{
    _data_frame.setContinuousLabel(label);
}

void GesticRecorder::onStopRecording()
{
    _is_recording = false;
    _record_till = QTime::currentTime();
    qDebug() << "GestIC record terminated";

    emit recordingStopped();
}

void GesticRecorder::onClearData()
{
    _data_frame.clear();
}


DataFrame::DataFrame() {
//    _label = LABELS::UNDEFINE;
//    _counter = 1;
    CURRENT_LABEL_ = LABELS::UNDEFINE;
    COUNTER_ = 1;
    clear();
}

void DataFrame::toCSV(QString file_name)
{
    if(_data_sd.size() > 0 && _data_cic.size() > 0){
        if(!file_name.endsWith("csv")){
            file_name += ".csv";
        }

        QFile file(file_name);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)){
            qDebug() << "Cannot open file";
            return;
        }

        QTextStream out(&file);
//        QDateTime now = QDateTime::currentDateTime();

        qDebug() << "No. " << COUNTER_ << " sample, Size: " << _data_sd.size();\


        // write headder
        out << '\t' << ','
            << TIME_STEMP_FOR_RECORD_.toString("yyyyMMddHHmmss") << ','
            << QString::number(COUNTER_) << ','
            << QString::number(CURRENT_LABEL_) << ','
            << QString::number(_data_sd.size()) << '\n';

        COUNTER_++;

        // write data
        if(_continuous_label.size() == 0){
            // sd_1, sd_2, sd_3, sd_4, sd_5, cic_1, cic_2, cic_3, cic_4, cic_5
            for(int i = 0; i < _data_sd.size(); i++){
                 out << _data_sd[i].channel[0] << ','
                     << _data_sd[i].channel[1] << ','
                     << _data_sd[i].channel[2] << ','
                     << _data_sd[i].channel[3] << ','
                     << _data_sd[i].channel[4] << ','
                     << _data_cic[i].channel[0] << ','
                     << _data_cic[i].channel[1] << ','
                     << _data_cic[i].channel[2] << ','
                     << _data_cic[i].channel[3] << ','
                     << _data_cic[i].channel[4] << '\n';
            }
        }else{
            // sd_1, sd_2, sd_3, sd_4, sd_5, cic_1, cic_2, cic_3, cic_4, cic_5, continuous_label
            for(int i = 0; i < _data_sd.size(); i++){
                 out << _data_sd[i].channel[0] << ','
                     << _data_sd[i].channel[1] << ','
                     << _data_sd[i].channel[2] << ','
                     << _data_sd[i].channel[3] << ','
                     << _data_sd[i].channel[4] << ','
                     << _data_cic[i].channel[0] << ','
                     << _data_cic[i].channel[1] << ','
                     << _data_cic[i].channel[2] << ','
                     << _data_cic[i].channel[3] << ','
                     << _data_cic[i].channel[4] << ','
                     << _continuous_label[i] << '\n';
            }
        }

        clear();

        file.close();

    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("No enough data to save!");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}

void DataFrame::clear()
{
    _data_sd.clear();
    _data_cic.clear();
    _continuous_label.clear();
}

void DataFrame::addData(gestic_signal_t data_sd, gestic_signal_t data_cic)
{
    _data_sd.append(data_sd);
    _data_cic.append(data_cic);
}

void DataFrame::setContinuousLabel(QList<double> label)
{
    _continuous_label = label;
}

QList<double> DataFrame::getContinuousLabel()
{
    return _continuous_label;
}

bool DataFrame::isEmpty()
{
    return _data_sd.isEmpty() || _data_cic.isEmpty() || _continuous_label.isEmpty();
}

int DataFrame::getLength()
{
    return _data_cic.size();
}

QList<gestic_signal_t> DataFrame::getData_cic()
{
    return _data_cic;
}

QList<gestic_signal_t> DataFrame::getData_sd()
{
    return _data_sd;
}
