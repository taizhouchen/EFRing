#include "linearslider.h"
#include "ui_linearslider.h"
#include <QTimer>
#include <QDateTime>
#include <QDebug>
#include <QtMath>
#include <QCloseEvent>

LinearSlider::LinearSlider(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::linearslider)
{
    ui->setupUi(this);

    _slider_width = 600;
    _slider_height = 100;

    _minimum = 0;
    _maximum = 10000;
    _single_step = 1;
    _current_pos = _minimum;
    _detector_status = DetectorLabel::OFF_RING;

    _label_offring = new QLabel(ui->linearSliderContainerWidget);
    _label_offring->setText("<html><head/><body><p><span style=\" font-size:48pt; font-weight:600; color:#2ecc71;\">OFF RING</span></p></body></html>");
    _label_offring->setAlignment(Qt::AlignCenter);

    _label_none = new QLabel(ui->linearSliderContainerWidget);
    _label_none->setText("<html><head/><body><p><span style=\" font-size:48pt; font-weight:600; color:#34495e;\">NO CONTACT</span></p></body></html>");
    _label_none->setAlignment(Qt::AlignCenter);

    _label_onring = new QLabel(ui->linearSliderContainerWidget);
    _label_onring->setText("<html><head/><body><p><span style=\" font-size:48pt; font-weight:600; color:#3498db;\">ON RING</span></p></body></html>");
    _label_onring->setAlignment(Qt::AlignCenter);

    _textInstruction = new QTextBrowser(this);
    _textInstruction->viewport()->setAutoFillBackground(false);
    _textInstruction->setText("Press [0~9] to control the moving speed \nPress [Space] to stop the cursr moving \nPress [r] to start recording \nPress [t] the start training \nPress [s] to save the model \nPress [c] to force calibrate (please move your thumb to the distal phalanx first before doing this)");

    QMenuBar *menu_bar = new QMenuBar(this);
    QMenu *tool_menu = menu_bar->addMenu("Tools");

    QAction *train_from_file_action = new QAction("Train From File", this);
    QAction *loadModel_action = new QAction("Load Model", this);
    QAction *loadDetectionModel_action = new QAction("Load Detection Model", this);

    tool_menu->addAction(loadModel_action);
    tool_menu->addAction(loadDetectionModel_action);
    tool_menu->addAction(train_from_file_action);

    connect(loadModel_action, &QAction::triggered, this, &LinearSlider::onLoadModelActionTriggered);
    connect(loadDetectionModel_action, &QAction::triggered, this, &LinearSlider::onLoadDetectionModelActionTriggered);
    connect(train_from_file_action, &QAction::triggered, this, &LinearSlider::onTrainFromFile);
}

LinearSlider::~LinearSlider()
{
    delete ui;
}

void LinearSlider::showEvent(QShowEvent*) {

//    setGeometry(0, 0, 1280, 720);
//    this->setFixedSize(geometry().width(), geometry().height());
//    this->setWindowState(Qt::WindowFullScreen);

    ui->linearSlider->setMinimum(_minimum);
    ui->linearSlider->setMaximum(_maximum);
    ui->linearSlider->setSingleStep(_single_step);
    ui->linearSlider->setValue(_current_pos);
    ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                  ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                  _slider_width,
                                  _slider_height);

    _current_slider_type = LinearSliderType::HORIZONTAL;

    ui->linearSlider->show();
    ui->linearSlider_user->hide();

    _y_h_max_min[0] = std::numeric_limits<double>::min();
    _y_h_max_min[1] = std::numeric_limits<double>::max();

    setSpeed(18);

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &LinearSlider::onTimerTimeOut);
    _timer->setTimerType(Qt::PreciseTimer);
    _timer->start(1);

    _exp_mov_avg = new ExpMovAvg(0.9);
    _mov_avg = new MovAvg(40);

    _is_recording = false;
    _slider_value.clear();

    _calibration_minimum = 0;
    _calibration_minimum_list.clear();
    _is_calibrating = false;

    _label_offring->move(width()/2 - _label_offring->width()/2, height()/4 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()/4 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()/4 - _label_onring->height()/2);

    _label_offring->hide();
    _label_none->hide();
    _label_onring->hide();

    setDefaultSVMParam();

    emit linearSliderReadySignal();
}

void LinearSlider::resizeEvent(QResizeEvent *event)
{
//    ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
//                                  ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
//                                  _slider_width,
//                                  _slider_height);
//    ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
//                                       ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
//                                       _slider_width,
//                                       _slider_height);



    if(_current_slider_type == LinearSliderType::VERTICAl) {
        ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_height/2,
                                      ui->linearSliderContainerWidget->height()/2 - _slider_width/2,
                                      _slider_height,
                                      _slider_width);

        ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_height/2,
                                           ui->linearSliderContainerWidget->height()/2 - _slider_width/2,
                                           _slider_height,
                                           _slider_width);

    }
    else if(_current_slider_type == LinearSliderType::HORIZONTAL) {
        ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                      ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                      _slider_width,
                                      _slider_height);

        ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                           ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                           _slider_width,
                                           _slider_height);

    }


    _label_offring->move(width()/2 - _label_offring->width()/2, height()/4 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()/4 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()/4 - _label_onring->height()/2);

    _textInstruction->setGeometry(0, 0, width(), height());
    _textInstruction->move(10, height()-100);
    _textInstruction->setFrameStyle(QFrame::NoFrame);
}

void LinearSlider::closeEvent(QCloseEvent*){
    _timer->stop();
    _current_pos = _minimum;
    disconnect(_timer, &QTimer::timeout, this, &LinearSlider::onTimerTimeOut);

    free(_param.weight_label);
    free(_param.weight);

//    free(_timer);
    free(_exp_mov_avg);
    free(_mov_avg);

    qDebug() << "linear slider window closing before emitting signal";
    emit linearSliderWidgetCloseSignal();
}

void LinearSlider::onTimerTimeOut()
{
    _current_pos += _speed;

    if(_last_pos % _maximum > _current_pos % _maximum){ // bound at maximun
        if(_last_timestemp != 0)
//            qDebug() << tr("Current Speed: %1").arg(QDateTime::currentMSecsSinceEpoch() - _last_timestemp);
        _last_timestemp = QDateTime::currentMSecsSinceEpoch();
    }

    if(_last_pos % (_maximum * 2) > _current_pos % (_maximum * 2)){    // bound at minimun
        if(_is_recording)
            stopRecording();
    }

    if(qFloor(_current_pos / _maximum) % 2 == 0){
        ui->linearSlider->setValue(_current_pos % _maximum);
    }
    else{
        ui->linearSlider->setValue(_maximum - (_current_pos % _maximum));
    }

    _last_pos = _current_pos;
}

void LinearSlider::onTrainingTimerOut()
{
    _timer->stop();
}

void LinearSlider::onGesticDataRecorded()
{
    _is_recording = true;
    double normed_value = double(ui->linearSlider->value()) / double(_maximum);
    _slider_value.append(normed_value);
}

void LinearSlider::onGesticRegressorSVMPredict(double y_h, double *y_h_max_min_trainingset)
{
    if(!ui->linearSlider_user->isVisible()) {
        ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                           ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                           _slider_width,
                                           _slider_height);

        ui->linearSlider_user->setMinimum(_minimum);
        ui->linearSlider_user->setMaximum(50);
        ui->linearSlider_user->setSingleStep(_single_step);
        ui->linearSlider_user->setValue(0);

        // disable the blue one
        _timer->stop();
        ui->linearSlider->hide();
        ui->linearSlider_user->show();
    }


    double value = y_h;

//    _exp_mov_avg->update(&value);   // exponential smoothing
    _mov_avg->update(&value);   // moving average

    if(_is_calibrating){
        if(_calibration_minimum_list.size() < 19){
            _calibration_minimum_list.append(y_h);
        }else{

            qSort(_calibration_minimum_list.begin(), _calibration_minimum_list.end());

            _calibration_minimum = _calibration_minimum_list[int(_calibration_minimum_list.size() / 2)]; // get the medium
            _is_calibrating = false;

            _y_h_max_min[1] = _calibration_minimum;
        }
    }

    if(value > _y_h_max_min[0]) _y_h_max_min[0] = value;
    if(value < _y_h_max_min[1]) _y_h_max_min[1] = value;

    map(&value, *(_y_h_max_min), *(_y_h_max_min + 1), (double)50, (double)_minimum);

//    if(value == _maximum) _y_h_max_min[1] = 0.3;
//    if(value == _minimum) _y_h_max_min[0] = 0.7;

    qDebug() << y_h << value << ": " << *(_y_h_max_min) << *(_y_h_max_min + 1);
    qDebug() << _detector_status;

    if(_detector_status == DetectorLabel::OFF_RING)
        ui->linearSlider_user->setValue((int)value);

//    _current_user_slider_value_normed += (y_h - 0.5);

//    double value = _current_user_slider_value_normed;

//    if(value > _y_h_max_min[0]) _y_h_max_min[0] = value;
//    if(value < _y_h_max_min[1]) _y_h_max_min[1] = value;

//    map(&value, _y_h_max_min[0], _y_h_max_min[1], (double)_maximun, (double)_minimum);

//    qDebug() << (int)value;
//     ui->linearSlider_user->setValue((int)value);

}

void LinearSlider::onGesticDetectorSVMPredict(double y_h, double *)
{
    switch (int(y_h)) {
        case DetectorLabel::NO_CONTACT:
            _label_offring->hide();
            _label_onring->hide();
            _label_none->show();
            _detector_status = DetectorLabel::NO_CONTACT;
            break;
        case DetectorLabel::OFF_RING:
            _label_onring->hide();
            _label_none->hide();
            _label_offring->show();
            _detector_status = DetectorLabel::OFF_RING;
            break;
        case DetectorLabel::ON_RING:
            _label_offring->hide();
            _label_none->hide();
            _label_onring->show();
            _detector_status = DetectorLabel::ON_RING;
            break;
    }
}

void LinearSlider::onLoadModelActionTriggered()
{
    emit linearSliderLoadModelSignal();
}

void LinearSlider::onLoadDetectionModelActionTriggered()
{
    emit linearSliderLoadDetectionModelSignal();
}

void LinearSlider::onTrainFromFile()
{
    QString file_name = QFileDialog::getOpenFileName(this, tr("Train from file"));
    if(file_name != ""){

        if(!file_name.endsWith(".csv")){
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.setText(QString("Invalided file %1").arg(file_name));
            msgBox.setWindowTitle("Message");
            msgBox.exec();

           return;
        }

        QList<SVMPoint> _svm_point_all;

        QFile file(file_name);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
                QMessageBox msgBox;
                msgBox.setIcon(QMessageBox::Critical);
                msgBox.setText(QString("Failed to load file %1").arg(file_name));
                msgBox.setWindowTitle("Message");
                msgBox.exec();

               return;
         }


        QTextStream in(&file);

        while (!in.atEnd()){
            QString line = in.readLine();
            if(line.startsWith('\t')) continue;
            QStringList split = line.split(',');

            if(split.length() != 11) continue;

            SVMPoint svm_point;

            svm_point.sd_c0 = split[0].toDouble();
            svm_point.sd_c1 = split[1].toDouble();
            svm_point.sd_c2 = split[2].toDouble();
            svm_point.sd_c3 = split[3].toDouble();
            svm_point.sd_c4 = split[4].toDouble();
            svm_point.cic_c0 = split[5].toDouble();
            svm_point.cic_c1 = split[6].toDouble();
            svm_point.cic_c2 = split[7].toDouble();
            svm_point.cic_c3 = split[8].toDouble();
            svm_point.cic_c4 = split[9].toDouble();

            svm_point.y = split[10].toDouble();

            _svm_point_all.append(svm_point);
        }

        if(_svm_point_all.size() > 0){
            setDefaultSVMParam();
            emit linearSliderStartTrainingSignalUsingSVMPoint(_svm_point_all, _param);
        }
    }
}

void LinearSlider::setSpeed(int speed){
    _speed = speed;
    qDebug() << tr("Speed: %1s").arg(float(_maximum - _minimum) / (float(_speed) * 1000));
}

void LinearSlider::keyPressEvent(QKeyEvent * event)
{
    switch (event->key()) {
        case Qt::Key_F1:
            this->isFullScreen() ? this->showNormal() : this->showFullScreen();
            break;
        case Qt::Key_1:
            setSpeed(2);
            break;
        case Qt::Key_2:
            setSpeed(4);
            break;
        case Qt::Key_3:
            setSpeed(6);
            break;
        case Qt::Key_4:
            setSpeed(8);
            break;
        case Qt::Key_5:
            setSpeed(10);
            break;
        case Qt::Key_6:
            setSpeed(12);
            break;
        case Qt::Key_7:
            setSpeed(14);
            break;
        case Qt::Key_8:
            setSpeed(16);
            break;
        case Qt::Key_9:
            setSpeed(18);
            break;
        case Qt::Key_0:
            setSpeed(20);
            break;
        case Qt::Key_R:
            if(!ui->linearSlider->isVisible()) return; // disable while predicting
//            _slider_value.clear();
            if(!_timer->isActive()){
                _current_pos = _minimum;
                ui->linearSlider->setValue(_current_pos);
                _timer->start(1);
            }
            emit linearSliderStartRecordSignal(_current_slider_type);
            break;
        case Qt::Key_S:
            if(!ui->linearSlider->isVisible()) return; // disable while predicting
            if(_slider_value.size() == 0){
                QMessageBox msgBox;
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.setText("No data to save");
                msgBox.setWindowTitle("Message");
                msgBox.exec();
            }else
                emit linearSliderSaveFileSignal(_slider_value);
            break;
        case Qt::Key_T:
            if(_slider_value.size() > 0){
                setDefaultSVMParam();
                emit linearSliderStartTrainingSignal(_slider_value, _param);
            }
            break;
        case Qt::Key_C:
            _y_h_max_min[0] = 0.7;
            _y_h_max_min[1] = 0.3;
//            _y_h_max_min[0] = std::numeric_limits<double>::min();
//            _y_h_max_min[1] = std::numeric_limits<double>::max();
            _exp_mov_avg->reset();
            _mov_avg->reset();

            _calibration_minimum_list.clear();
            _is_calibrating = true;

            emit forceCalibrateSignal();
            break;
        case Qt::Key_Space:
            if(_timer->isActive()){
                _timer->stop();
                _current_pos = _minimum;
                ui->linearSlider->setValue(_current_pos);
                _last_pos = 0;
            }else{
                _timer->start(1);
            }
            break;
        case Qt::Key_V:
            if(_current_slider_type == LinearSliderType::HORIZONTAL) {
                ui->linearSlider->setOrientation(Qt::Vertical);
                ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_height/2,
                                              ui->linearSliderContainerWidget->height()/2 - _slider_width/2,
                                              _slider_height,
                                              _slider_width);

                ui->linearSlider_user->setOrientation(Qt::Vertical);
                ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_height/2,
                                                   ui->linearSliderContainerWidget->height()/2 - _slider_width/2,
                                                   _slider_height,
                                                   _slider_width);

                _current_slider_type = LinearSliderType::VERTICAl;
            }
            else if(_current_slider_type == LinearSliderType::VERTICAl) {
                ui->linearSlider->setOrientation(Qt::Horizontal);
                ui->linearSlider->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                              ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                              _slider_width,
                                              _slider_height);

                ui->linearSlider_user->setOrientation(Qt::Horizontal);
                ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                                   ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                                   _slider_width,
                                                   _slider_height);

                _current_slider_type = LinearSliderType::HORIZONTAL;
            }
            break;
        case Qt::Key_Escape:
            this->close();
            break;
    }
}


void LinearSlider::stopRecording()
{
    _timer->stop();
    _is_recording = false;
    emit stopRecordingSignal();
    _current_pos = _minimum;
    ui->linearSlider->setValue(_current_pos);
}

void LinearSlider::setDefaultSVMParam()
{
    // default values
    _param.svm_type = EPSILON_SVR;
    _param.kernel_type = RBF;
    _param.degree = 3;
    _param.gamma = 1/10;	// 1/num_features
    _param.coef0 = 0;
    _param.nu = 0.5;
    _param.cache_size = 100;
    _param.C = 1;
    _param.eps = 1e-3;
    _param.p = 0.1;
    _param.shrinking = 1;
    _param.probability = 0;
    _param.nr_weight = 0;
    _param.weight_label = NULL;
    _param.weight = NULL;
}

void LinearSlider::map(double *v, double v_max, double v_min, double target_max, double target_min)
{
    *v = ((*v) - v_min) / (v_max - v_min) * (target_max - target_min) + target_min;
}
