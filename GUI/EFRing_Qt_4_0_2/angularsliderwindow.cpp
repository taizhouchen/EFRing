#include "angularsliderwindow.h"

AngularSliderWindow::AngularSliderWindow(QWidget *parent)  : QWidget(parent)
{
    _angularSlider = new AngularSlider(QColor("#3498db"), this);
    _angularSlider_user = new AngularSlider(QColor("#c0392b"), this);
}

void AngularSliderWindow::setSpeed(double s)
{
    _speed = s;
    qDebug() << tr("Speed: %1s").arg(360.0 / (float(_speed) * 1000));
}

void AngularSliderWindow::setDefaultSVMParam()
{
    // default values
    _param.svm_type = EPSILON_SVR;
    _param.kernel_type = RBF;
    _param.degree = 3;
    _param.gamma = 0.2;	// 1/num_features
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

void AngularSliderWindow::resizeEvent(QResizeEvent *event)
{
    _angularSlider->move(width()/2 - (_angularSlider->width()/2), height()/2 - (_angularSlider->height()/2));
    _angularSlider_user->move(width()/2 - (_angularSlider_user->width()/2), height()/2 - (_angularSlider_user->height()/2));
}

void AngularSliderWindow::showEvent(QShowEvent *event)
{
    setWindowTitle(tr("Angular Slider"));
//    resize(1280, 720);
//    this->setFixedSize(geometry().width(), geometry().height());
//    this->setWindowState(Qt::WindowFullScreen);

    _current_angle_slider = 0;
    _current_angle_slider_user = 0;
    _angularSlider->setAngle(_current_angle_slider);
    _angularSlider_user->setAngle(_current_angle_slider_user);

    _angularSlider->show();
    _angularSlider_user->hide();

    _y_h_max_min[0] = std::numeric_limits<double>::min();
    _y_h_max_min[1] = std::numeric_limits<double>::max();

    setSpeed(0.1);

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &AngularSliderWindow::onTimerTimeOut);
    _timer->setTimerType(Qt::PreciseTimer);
    _timer->start(1);

    _exp_mov_avg = new ExpMovAvg(0.9);
    _mov_avg = new MovAvg(20);

    _is_recording = false;

    setDefaultSVMParam();

    emit angularSliderReadySignal();
}

void AngularSliderWindow::closeEvent(QCloseEvent *event)
{
    _timer->stop();
    disconnect(_timer, &QTimer::timeout, this, &AngularSliderWindow::onTimerTimeOut);
    _current_angle_slider = 0;
    _current_angle_slider_user = 0;

    free(_param.weight_label);
    free(_param.weight);

    free(_exp_mov_avg);
    free(_mov_avg);

    emit angularSliderWidgetCloseSignal();
}

void AngularSliderWindow::keyPressEvent(QKeyEvent *event)
{
    switch(event->key()){
        case Qt::Key_1:
            setSpeed(0.1);
            break;
        case Qt::Key_2:
            setSpeed(0.2);
            break;
        case Qt::Key_3:
            setSpeed(0.3);
            break;
        case Qt::Key_4:
            setSpeed(0.4);
            break;
        case Qt::Key_5:
            setSpeed(0.5);
            break;
        case Qt::Key_6:
            setSpeed(0.6);
            break;
        case Qt::Key_7:
            setSpeed(0.7);
            break;
        case Qt::Key_8:
            setSpeed(0.8);
            break;
        case Qt::Key_9:
            setSpeed(0.9);
            break;
        case Qt::Key_0:
            setSpeed(1);
            break;
        case Qt::Key_R:
            if(!_angularSlider->isVisible()) return;    // disable while predicting
            _slider_value.clear();
            if(!_timer->isActive()){
                _current_angle_slider = 0;
                _angularSlider->setAngle(_current_angle_slider);
                _timer->start(1);
            }

            emit angularSliderStartRecordSignal();
            break;
        case Qt::Key_S:
            if(!_angularSlider->isVisible()) return;
            if(_slider_value.size() == 0){
                QMessageBox msgBox;
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.setText("No data to save");
                msgBox.setWindowTitle("Message");
                msgBox.exec();
            }else
                emit angularSliderSaveFileSignal(_slider_value);
            break;
        case Qt::Key_T:
            if(_slider_value.size() > 0){
                setDefaultSVMParam();
                emit angularSliderStartTrainingSignal(_slider_value, _param);
            }
            break;
        case Qt::Key_Space:
            if(_timer->isActive()){
                _timer->stop();
                _current_angle_slider = 0;
                _angularSlider->setAngle(_current_angle_slider);
            }else{
                _timer->start(1);
            }
            break;
        case Qt::Key_C:
            _y_h_max_min[0] = 0.7;
            _y_h_max_min[1] = 0.3;
//            _y_h_max_min[0] = std::numeric_limits<double>::min();
//            _y_h_max_min[1] = std::numeric_limits<double>::max();
            _exp_mov_avg->reset();
            _mov_avg->reset();

            emit forceCalibrateSignal();
            break;
        case Qt::Key_Escape:
            this->close();
            break;
    }
}

void AngularSliderWindow::onTimerTimeOut()
{
    _current_angle_slider += _speed;
    if(_current_angle_slider > 360) {
        if(_is_recording) stopRecording();
        else _current_angle_slider = 0;
    }

    _angularSlider->setAngle(_current_angle_slider);
}

void AngularSliderWindow::onTrainingTimerOut(){
    _timer->stop();
}

void AngularSliderWindow::onGesticDataRecorded(){
    _is_recording = true;
    double normed_value = double(_angularSlider->getAngle()) / double(360.0);
    _slider_value.append(normed_value);
}

void AngularSliderWindow::onGesticRegressorSVMPredict(double y_h, double *y_h_max_min_trainingset){
    if(!_angularSlider_user->isVisible()){
        _angularSlider_user->setAngle(0);

        // disable the blue one
        _timer->stop();
        _angularSlider->hide();
        _angularSlider_user->show();
    }

    double value = y_h;

//    _exp_mov_avg->update(&value);   // exponential smoothing
    _mov_avg->update(&value);   // moving average

    if(value > _y_h_max_min[0]) _y_h_max_min[0] = value;
    if(value < _y_h_max_min[1]) _y_h_max_min[1] = value;

    map(&value, *(_y_h_max_min), *(_y_h_max_min + 1), (double)360.0, (double)0.0);
    qDebug() << y_h << value << ": " << *(_y_h_max_min) << *(_y_h_max_min + 1);

    _angularSlider_user->setAngle(value);
}

void AngularSliderWindow::map(double *v, double v_max, double v_min, double target_max, double target_min){
    *v = ((*v) - v_min) / (v_max - v_min) * (target_max - target_min) + target_min;
}

void AngularSliderWindow::stopRecording()
{
    _timer->stop();
    _is_recording = false;
    emit stopRecordingSignal();
    _current_angle_slider = 359.9999;
    _angularSlider->setAngle(_current_angle_slider);
}






AngularSlider::AngularSlider(QColor c, QWidget *parent)
    : QWidget(parent)
{
    resize(1280, 720);
    _color = c;
    _current_angle = 0;
}


void AngularSlider::setAngle(double a)
{
    _current_angle = a;

    update();
}

void AngularSlider::setColor(QColor c)
{
    _color = c;
}

double AngularSlider::getAngle()
{
    return _current_angle;
}

void AngularSlider::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    painter.setRenderHint(QPainter::Antialiasing);
    painter.translate(width()/2, height()/2);

    painter.rotate(_current_angle);
    painter.drawLine(0, 0, 200, 0);

    QRectF rectangle(200, 0, 50, 50);
    rectangle.moveCenter(QPointF(200, 0));

    QPainterPath path;
    path.addRoundedRect(rectangle, 10, 10);
    QPen pen(Qt::black, 1);
    painter.setPen(pen);
    painter.fillPath(path, _color);
    painter.drawPath(path);
}

