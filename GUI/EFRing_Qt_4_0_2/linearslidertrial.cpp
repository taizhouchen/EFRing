#include "linearslidertrial.h"
#include "ui_linearslider.h"

LinearSliderTrial::LinearSliderTrial(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::linearslider)
{
    ui->setupUi(this);

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

    _label_indicator = new QLabel(ui->linearSliderContainerWidget);

    _textInstruction = new QTextBrowser(this);
    _textInstruction->viewport()->setAutoFillBackground(false);
    _textInstruction->setText("Press [space] to start the trial \nPress [r] to restart the trial \nPress [f] to switch between different smoother \nPress [c] to force calibrate (please move your thumb to the distal phalanx first before doing this)");

    QMenuBar *menu_bar = new QMenuBar(this);
    QMenu *tool_menu = menu_bar->addMenu("Tools");

    QAction *loadModel_action = new QAction("Load Model", this);
    QAction *loadDetectionModel_action = new QAction("Load Detection Model", this);

    tool_menu->addAction(loadModel_action);
    tool_menu->addAction(loadDetectionModel_action);

    connect(loadModel_action, &QAction::triggered, this, &LinearSliderTrial::onLoadModelActionTriggered);
    connect(loadDetectionModel_action, &QAction::triggered, this, &LinearSliderTrial::onLoadDetectionModelActionTriggered);
}

LinearSliderTrial::~LinearSliderTrial()
{
    delete ui;
}

void LinearSliderTrial::setFileName(QString file_name)
{
    _file_name = file_name;
}

void LinearSliderTrial::showEvent(QShowEvent *)
{
    ui->linearSlider->hide();
    ui->linearSlider_user->show();

    ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                       ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                       _slider_width,
                                       _slider_height);

    ui->linearSlider_user->setMinimum(_minimum);
    ui->linearSlider_user->setMaximum(_maximum);
    ui->linearSlider_user->setSingleStep(_single_step);
    ui->linearSlider_user->setValue(0);

    connect(ui->linearSlider_user, &QSlider::valueChanged, this, &LinearSliderTrial::onSliderValueChanged);

    _fittslaw_problem_list.clear();
    _calibration_minimum_list.clear();
    _prediction_buffer.clear();

    buildFittsLawTrial();

    _y_h_max_min[0] = std::numeric_limits<double>::min();
    _y_h_max_min[1] = std::numeric_limits<double>::max();

//    _exp_mov_avg = new ExpMovAvg(0.9);
    _mov_avg = new MovAvg(50);
    _kf = new KF();

    _label_offring->move(width()/2 - _label_offring->width()/2, height()*3/4 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()*3/4 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()*3/4 - _label_onring->height()/2);

    _label_indicator->setGeometry(0, 0, width(), height()/4);
    _label_indicator->move(width()/2 - _label_indicator->width()/2, height()/8 - _label_onring->height()/2);
    _label_indicator->setAlignment(Qt::AlignCenter);
    _label_indicator->setText("");

    _label_offring->hide();
    _label_none->hide();
    _label_onring->hide();

    _using_filter = Filters::KalmanFilter;
    _trial_started = false;
    ui->linearSlider_user->setStyleSheet(tr("QSlider::groove:horizontal {height: 8px;background: #7f8c8d;margin: 2px 0;}QSlider::handle:horizontal {background: #95a5a6;border: 1px solid #5c5c5c;	width: 20px;margin: -20px 0px;border-radius:  10px;}"));

    emit linearSliderReadySignal();
}

void LinearSliderTrial::closeEvent(QCloseEvent *)
{
    free(_exp_mov_avg);
    free(_mov_avg);
    free(_kf);

    qDebug() << "linear slider window closing before emitting signal";
    emit linearSliderWidgetCloseSignal();
}

void LinearSliderTrial::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_F1:
            this->isFullScreen() ? this->showNormal() : this->showFullScreen();
            break;
        case Qt::Key_Space:
            nextTrial();
            break;
        case Qt::Key_R:
            _current_trial_index = -1;
            _ready_for_next_trial = true;
            nextTrial();
            break;
        case Qt::Key_C:
            _y_h_max_min[0] = 0.9;
            _y_h_max_min[1] = 0.3;

            switch (_using_filter) {
                case Filters::KalmanFilter:
                    break;
                case Filters::MovingAverage:
                    _mov_avg->reset();
                    break;
            }

            _calibration_minimum_list.clear();
            _is_calibrating = true;


            emit forceCalibrateSignal();
            break;
        case Qt::Key_Escape:
            this->close();
            break;
        case Qt::Key_F:
            switch (_using_filter) {
                case Filters::KalmanFilter:
                    _using_filter = Filters::MovingAverage;
                    qDebug() << "Using Moving Average";
                    break;
                case Filters::MovingAverage:
                    _using_filter = Filters::KalmanFilter;
                    qDebug() << "Using Kalman Filter";
                    break;
                    }
              break;
         case Qt::Key_K:
                switch (_using_filter) {
                    case Filters::KalmanFilter:
                        _kf->reset();
                        qDebug() << "Kalman Filter Reseted";
                        break;
                    case Filters::MovingAverage:
                        break;
                }
                break;
    }
}

void LinearSliderTrial::resizeEvent(QResizeEvent *)
{
    ui->linearSlider_user->setGeometry(ui->linearSliderContainerWidget->width()/2 - _slider_width/2,
                                       ui->linearSliderContainerWidget->height()/2 - _slider_height/2,
                                       _slider_width,
                                       _slider_height);

    _label_offring->move(width()/2 - _label_offring->width()/2, height()*3/4 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()*3/4 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()*3/4 - _label_onring->height()/2);

    _label_indicator->setGeometry(0, 0, width(), height()/4);
    _label_indicator->move(width()/2 - _label_indicator->width()/2, height()/8 - _label_onring->height()/2);

    _textInstruction->setGeometry(0, 0, width(), height());
    _textInstruction->move(10, height()-100);
    _textInstruction->setFrameStyle(QFrame::NoFrame);
}

void LinearSliderTrial::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(QRectF(0, 0, width(), height()), Qt::white);

    painter.translate(width()/2-_slider_width/2, height()/2);

    if(_current_trial_index >= 0){
        int x, y, w, h;
        h = 120;
//        x = _fittslaw_problem_list[_current_trial_index].D_pixel - _fittslaw_problem_list[_current_trial_index].W/2 + _handle_width/2;
        x = _fittslaw_problem_list[_current_trial_index].D_pixel - _fittslaw_problem_list[_current_trial_index].W/2;
        y = 0 - h/2;
        w = _fittslaw_problem_list[_current_trial_index].W;
        QRectF rectangle(x, y, w, h);

        QPainterPath path;
        path.addRoundedRect(rectangle, 10, 10);
        QPen pen(Qt::black, 2);
        painter.setPen(pen);
        painter.drawPath(path);

//        int range_upper = _fittslaw_problem_list[_current_trial_index].D_pixel + _fittslaw_problem_list[_current_trial_index].W/2;
//        int range_lower = _fittslaw_problem_list[_current_trial_index].D_pixel - _fittslaw_problem_list[_current_trial_index].W/2;
//        painter.drawLine(range_lower, -100, range_lower, 100);
//        painter.drawLine(range_upper, -100, range_upper, 100);
//        painter.drawText(QPointF(range_lower, -100), QString::number(range_lower));
//        painter.drawText(QPointF(range_upper, -100), QString::number(range_upper));

//        int current_pixel = 0;
//        valueToPixel(ui->linearSlider_user->value(), &current_pixel);
//        painter.drawText(QPointF(current_pixel, 50), QString::number(current_pixel));

        if(_fill)
            painter.fillPath(path, QColor("#e74c3c"));
    }
}

void LinearSliderTrial::valueToPixel(int value, int *pixel)
{
    double *value_out;
    double a = 1.0;
    value_out = &a;
    *value_out = value;
    map(value_out, _maximum, _minimum, _slider_width-_handle_width/2, _handle_width/2);
//    map(value_out, _maximum, _minimum, _slider_width-_slider_margin, _slider_margin);
    *pixel = (int)*value_out;
}

void LinearSliderTrial::pixelToValue(int pixel, int *value)
{
    double *pixel_out;
    double a = 1.0;
    pixel_out = &a;
    *pixel_out = pixel;
    map(pixel_out, _slider_width-_handle_width/2, _handle_width/2, _maximum, _minimum);
//    map(pixel_out, _slider_width-_slider_margin, _slider_margin, _maximum, _minimum);
    *value = (int)*pixel_out;
}

void LinearSliderTrial::map(double *v, double v_max, double v_min, double target_max, double target_min)
{
    *v = ((*v) - v_min) / (v_max - v_min) * (target_max - target_min) + target_min;
}

void LinearSliderTrial::buildFittsLawTrial()
{
    for(int r = 0; r < _repeat; r++){
        for(int d = 0; d < D.size(); d++){
            for(int w = 0; w < W.size(); w++){

                fittslaw_problem_node node = fittslaw_problem_node();
                int pixel = 0;
                valueToPixel(D[d], &pixel);
                node.D = D[d];
                node.W = W[w];
                node.D_pixel = pixel;
                node.handle_width = _handle_width;

                node.ID = log2(double(pixel)/double(W[w]) + double(1.0));

                _fittslaw_problem_list.append(node);
            }
        }
    }
    std::random_shuffle(_fittslaw_problem_list.begin(), _fittslaw_problem_list.end());

    _current_trial_index = -1;
    _ready_for_next_trial = true;
}

void LinearSliderTrial::nextTrial()
{
    if(_ready_for_next_trial){
        if(_current_trial_index != -1){
            // save data for current trial
            _fittslaw_problem_list[_current_trial_index].T = _trial_end_time - _trial_start_time;
            _fittslaw_problem_list[_current_trial_index].distance = _integrated_distance;
            _fittslaw_problem_list[_current_trial_index].last_pos = _prev_pos;

            qDebug() << "D: " << _fittslaw_problem_list[_current_trial_index].D;
            qDebug() << "W: " << _fittslaw_problem_list[_current_trial_index].W;
            qDebug() << "T: " << _fittslaw_problem_list[_current_trial_index].T;
            qDebug() << "ID: " << _fittslaw_problem_list[_current_trial_index].ID;
            qDebug() << "D_pixel: " << _fittslaw_problem_list[_current_trial_index].D_pixel;
            qDebug() << "Total Distance: " << _fittslaw_problem_list[_current_trial_index].distance;
            qDebug() << "Last Pos: " << _fittslaw_problem_list[_current_trial_index].last_pos;
        }

        // move to next trial
        if(_current_trial_index == _fittslaw_problem_list.size() - 1){
            _current_trial_index = -1;
            _label_indicator->setText(tr("<html><head/><body><p><span style=\" font-size:70pt; font-weight:600; color:#34495e;\">END</span></p></body></html>"));
            saveToFile(_file_name);
        }else{
            _current_trial_index++;
            _fill = false;
            _ready_for_next_trial = false;
            _label_indicator->setText(tr("<html><head/><body><p><span style=\" font-size:70pt; font-weight:600; color:#34495e;\">(%1/%2) Next</span></p></body></html>").arg(_current_trial_index+1).arg(_fittslaw_problem_list.size()));
            _trial_started = false;
            ui->linearSlider_user->setStyleSheet(tr("QSlider::groove:horizontal {height: 8px;background: #7f8c8d;margin: 2px 0;}QSlider::handle:horizontal {background: #95a5a6;border: 1px solid #5c5c5c;	width: 20px;margin: -20px 0px;border-radius:  10px;}"));
        }

        update();
    }
}

void LinearSliderTrial::startTrial()
{
    // trial will be started after calibration
    if(_current_trial_index >= 0){
        _label_indicator->setText(tr("<html><head/><body><p><span style=\" font-size:70pt; font-weight:600; color:#1abc9c;\">(%1/%2) Start</span></p></body></html>").arg(_current_trial_index+1).arg(_fittslaw_problem_list.size()));
        _trial_start_time = QDateTime::currentMSecsSinceEpoch();
        _prev_pos = 0;
        _integrated_distance = 0;
        _ready_for_next_trial = false;
        _trial_started = true;
    }
}

void LinearSliderTrial::saveToFile(QString file_name)
{
    if(_fittslaw_problem_list.size() > 0 && file_name != ""){
        if(!file_name.endsWith("csv")){
            file_name += ".csv";
        }

        QFile file(file_name);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)){
            qDebug() << "Cannot open file";
            return;
        }

        QTextStream out(&file);
        int i = 0;

        out << "T,W,D_pixel,ID,D,distance,last_pos,handle_width\n";
        for(QList<fittslaw_problem_node>::Iterator p = _fittslaw_problem_list.begin(); p != _fittslaw_problem_list.end(); p++){
            out << p->T << ','
                << p->W << ','
                << p->D_pixel << ','
                << p->ID << ','
                << p->D << ','
                << p->distance << ','
                << p->last_pos << ','
                << p->handle_width << '\n';
            i++;
        }

        file.close();

        qDebug() << QString("%1 lines of data were saved to %2").arg(i).arg(file_name);
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("No enough data to save!");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}


void LinearSliderTrial::onSliderValueChanged()
{
    if(_current_trial_index >= 0){
        int current_pixel = 0;
        valueToPixel(ui->linearSlider_user->value(), &current_pixel);
        int range_upper = _fittslaw_problem_list[_current_trial_index].D_pixel + _fittslaw_problem_list[_current_trial_index].W/2;
        int range_lower = _fittslaw_problem_list[_current_trial_index].D_pixel - _fittslaw_problem_list[_current_trial_index].W/2;
        if(current_pixel >= range_lower && current_pixel <= range_upper){
            _trial_end_time = QDateTime::currentMSecsSinceEpoch();
            _fill = true;
            _ready_for_next_trial = true;
        }
        else{
            _fill = false;
            _ready_for_next_trial = false;
        }
        update();
    }
}

void LinearSliderTrial::onLoadModelActionTriggered()
{
    emit linearSliderLoadModelSignal();
}

void LinearSliderTrial::onLoadDetectionModelActionTriggered()
{
    emit linearSliderLoadDetectionModelSignal();
}

void LinearSliderTrial::onGesticRegressorSVMPredict(double y_h, double *)
{
    double value = y_h;

//    _exp_mov_avg->update(&value);   // exponential smoothing

    switch (_using_filter) {
        case Filters::KalmanFilter:
            _kf->update(&value);
            break;
        case Filters::MovingAverage:
            _mov_avg->update(&value);
            break;
    }
//    _kf->update(&value);    // kalman filter
//    _mov_avg->update(&value);   // moving average

    if(_is_calibrating){
        ui->linearSlider_user->setStyleSheet(tr("QSlider::groove:horizontal {height: 8px;background: #7f8c8d;margin: 2px 0;}QSlider::handle:horizontal {background: #95a5a6;border: 1px solid #5c5c5c;	width: 20px;margin: -20px 0px;border-radius:  10px;}"));
        if(_current_trial_index >= 0)
            _label_indicator->setText(tr("<html><head/><body><p><span style=\" font-size:70pt; font-weight:600; color:#34495e;\">(%1/%2) Calibrating</span></p></body></html>").arg(_current_trial_index+1).arg(_fittslaw_problem_list.size()));
        if(_calibration_minimum_list.size() < 190){
            _calibration_minimum_list.append(y_h);
        }else{

//            qSort(_calibration_minimum_list.begin(), _calibration_minimum_list.end());

//            _calibration_minimum = _calibration_minimum_list[int(_calibration_minimum_list.size() / 2)]; // get the medium

            _calibration_minimum = _calibration_minimum_list[_calibration_minimum_list.size()-1];
            _is_calibrating = false;

            _y_h_max_min[1] = _calibration_minimum;
            _y_h_max_min[0] = _calibration_minimum + 0.7;

            ui->linearSlider_user->setStyleSheet(tr("QSlider::groove:horizontal {height: 8px;background: #7f8c8d;margin: 2px 0;}QSlider::handle:horizontal {background: #c0392b;border: 1px solid #5c5c5c;	width: 20px;margin: -20px 0px;border-radius:  10px;}"));
            startTrial();
        }
    }else if(_trial_started){

        if(value > _y_h_max_min[0]) _y_h_max_min[0] = value;
        if(value < _y_h_max_min[1]) _y_h_max_min[1] = value;

        map(&value, *(_y_h_max_min), *(_y_h_max_min + 1), (double)_maximum, (double)_minimum);



    //    qDebug() << y_h << value << ": " << *(_y_h_max_min) << *(_y_h_max_min + 1);



        _prediction_buffer.push_back(value);

        if(_detector_status == DetectorLabel::OFF_RING && _prediction_buffer.size() >= _prediction_buffer_size){
    //        ui->linearSlider_user->setValue((int)value);
            ui->linearSlider_user->setValue((int)_prediction_buffer[0]);

            // calculate total moving distance
            _integrated_distance += abs((int)_prediction_buffer[0] - _prev_pos);
            _prev_pos = (int)_prediction_buffer[0];
        }

        if(_prediction_buffer.size() > _prediction_buffer_size)
            _prediction_buffer.pop_front();
    }

}

void LinearSliderTrial::onGesticDetectorSVMPredict(double y_h, double *)
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
