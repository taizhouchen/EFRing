#include "tracker2dwindow.h"

Tracker2DWindow::Tracker2DWindow(QWidget *parent) : QWidget(parent)
{
    _anchor = new AnchorWidget(QColor("#3498db"), this);
    _anchor_user = new AnchorWidget(QColor("#c0392b"), this);

    QMenuBar *menu_bar = new QMenuBar(this);
    QMenu *tool_menu = menu_bar->addMenu("Tools");

    QAction *loadModelX_action = new QAction("Load Model x", this);
    QAction *loadModelY_action = new QAction("Load Model y", this);
    QAction *loadDetectionModel_action = new QAction("Load Detection Model", this);


    tool_menu->addAction(loadModelX_action);
    tool_menu->addAction(loadModelY_action);
    tool_menu->addAction(loadDetectionModel_action);

    connect(loadModelX_action, &QAction::triggered, this, &Tracker2DWindow::onLoadModelXActionTriggered);
    connect(loadModelY_action, &QAction::triggered, this, &Tracker2DWindow::onLoadModelYActionTriggered);
    connect(loadDetectionModel_action, &QAction::triggered, this, &Tracker2DWindow::onLoadDetectionModelActionTriggered);
}

void Tracker2DWindow::resizeEvent(QResizeEvent *event)
{
    _anchor->move(width()/2 - (_anchor->width()/2), height()/2 - (_anchor->height()/2));
    _anchor_user->move(width()/2 - (_anchor_user->width()/2), height()/2 - (_anchor_user->height()/2));
}

void Tracker2DWindow::showEvent(QShowEvent *event)
{
    setWindowTitle(tr("2d Tracker"));
    resize(1280, 720);

    _speed_x = 0.0018;
    _speed_y = 0.0018;
    _last_pos_x = 0;
    _last_pos_y = 1;

    _anchor->setPos(0, 1);
    _anchor_user->setPos(0, 1);

    _anchor->show();
    _anchor_user->hide();

    _y_h_x_max_min[0] = std::numeric_limits<double>::min();
    _y_h_x_max_min[1] = std::numeric_limits<double>::max();
    _y_h_y_max_min[0] = std::numeric_limits<double>::min();
    _y_h_y_max_min[1] = std::numeric_limits<double>::max();

    setDefaultSVMParamX();
    setDefaultSVMParamY();

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &Tracker2DWindow::onTimerTimeOut);
    _timer->setTimerType(Qt::PreciseTimer);
    _timer->start(1);

    _is_recording = false;

    emit tracker2dWindowReadySignal();
}

void Tracker2DWindow::closeEvent(QCloseEvent *event)
{
    _timer->stop();
    disconnect(_timer, &QTimer::timeout, this, &Tracker2DWindow::onTimerTimeOut);

    free(_param_x.weight_label);
    free(_param_x.weight);
    free(_param_y.weight_label);
    free(_param_y.weight);

    emit tracker2dWindowCloseSignal();
}

void Tracker2DWindow::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_R:
            if(!_anchor->isVisible()) return;
//            _pos_x_list.clear();
//            _pos_y_list.clear();
            if(!_timer->isActive()){
                _anchor->setPos(0, 1);
                _timer->start();
                _is_recording = true;
            }
            emit tracker2dWindowStartRecordSignal();
        break;

        case Qt::Key_T:
            if(_pos_x_list.size() > 0 && _pos_y_list.size() > 0){
                setDefaultSVMParamX();
                setDefaultSVMParamY();

                emit tracker2dWindowStartTrainingSignal(_pos_x_list, _pos_y_list, _param_x, _param_y);
            }
        break;

        case Qt::Key_C:
            _y_h_x_max_min[0] = 0.7;
            _y_h_x_max_min[1] = 0.3;
            _y_h_y_max_min[0] = 0.7;
            _y_h_y_max_min[1] = 0.3;

            emit forceCalibrateSignal();
        break;

        case Qt::Key_Space:
            if(_timer->isActive()){
                _timer->stop();
                _anchor->setPos(0, 1);

                _last_pos_x = _anchor->getPos().x();
                _last_pos_y = _anchor->getPos().y();

            }else{
                _timer->start(1);
            }
        break;

        case Qt::Key_Escape:
            this->close();
        break;
    }
}

void Tracker2DWindow::setDefaultSVMParamX()
{
    // default values
    _param_x.svm_type = EPSILON_SVR;
    _param_x.kernel_type = RBF;
    _param_x.degree = 3;
    _param_x.gamma = 0.2;	// 1/num_features
    _param_x.coef0 = 0;
    _param_x.nu = 0.5;
    _param_x.cache_size = 100;
    _param_x.C = 1;
    _param_x.eps = 1e-3;
    _param_x.p = 0.1;
    _param_x.shrinking = 1;
    _param_x.probability = 0;
    _param_x.nr_weight = 0;
    _param_x.weight_label = NULL;
    _param_x.weight = NULL;
}

void Tracker2DWindow::setDefaultSVMParamY()
{
    // default values
    _param_y.svm_type = EPSILON_SVR;
    _param_y.kernel_type = RBF;
    _param_y.degree = 3;
    _param_y.gamma = 0.2;	// 1/num_features
    _param_y.coef0 = 0;
    _param_y.nu = 0.5;
    _param_y.cache_size = 100;
    _param_y.C = 1;
    _param_y.eps = 1e-3;
    _param_y.p = 0.1;
    _param_y.shrinking = 1;
    _param_y.probability = 0;
    _param_y.nr_weight = 0;
    _param_y.weight_label = NULL;
    _param_y.weight = NULL;
}

void Tracker2DWindow::stopRecording()
{
    _timer->stop();
    _is_recording = false;
    emit stopRecordingSignal();
    _anchor->setPos(0, 1);
}

void Tracker2DWindow::map(double *v, double v_max, double v_min, double target_max, double target_min){
    *v = ((*v) - v_min) / (v_max - v_min) * (target_max - target_min) + target_min;
}

void Tracker2DWindow::onTimerTimeOut()
{
//    if(_anchor->getPos().x() >= 1. || _anchor->getPos().x() <= 0.)
//        _speed_x = _speed_x * -1.;
//    if(_anchor->getPos().y() >= 1. || _anchor->getPos().y() <= 0.)
//        _speed_y = _speed_y * -1.;

    if(_anchor->getPos().x() <= 0. && _anchor->getPos().y() >= 1.)    // bottom left
    {
        dx_ = 0;
        dy_ = _speed_y * -1.;
    }
    if(_anchor->getPos().x() <= 0. && _anchor->getPos().y() <= 0.)  // top left
    {
        dx_ = _speed_x;
        dy_ = 0;
    }
    if(_anchor->getPos().x() >= 1. && _anchor->getPos().y() >= 0.)  // top right
    {
        dx_ = 0;
        dy_ = _speed_y;
    }
    if(_anchor->getPos().x() >= 1. && _anchor->getPos().y() >= 1.)  // bottom right
    {
        dx_ = _speed_x * -1.;
        dy_ = 0;
    }

    _anchor->setPos(_anchor->getPos().x() + dx_, _anchor->getPos().y() + dy_);

    if(_is_recording){
        if(_anchor->getPos().x() == 0){
            if(_last_pos_x > _anchor->getPos().x()) {
                stopRecording();}
        }
    }

    _last_pos_x = _anchor->getPos().x();
    _last_pos_y = _anchor->getPos().y();
}

void Tracker2DWindow::onLoadModelXActionTriggered()
{
    emit tracker2dLoadModelXSignal();
}

void Tracker2DWindow::onLoadModelYActionTriggered()
{
    emit tracker2dLoadModelYSignal();
}

void Tracker2DWindow::onLoadDetectionModelActionTriggered()
{
    emit tracker2dLoadDetectionModelSignal();
}

void Tracker2DWindow::onGesticDataRecorded()
{
    _is_recording = true;
    _pos_x_list.append(_anchor->getPos().x());
    _pos_y_list.append(_anchor->getPos().y());
}

void Tracker2DWindow::onGesticRegressorSVMPredictX(double y_h, double *)
{


    if(!_anchor_user->isVisible()){
        _anchor_user->setPos(0, 1);

        _timer->stop();
        _anchor->hide();
        _anchor_user->show();
    }

    double value = y_h;

    if(value > _y_h_x_max_min[0]) _y_h_x_max_min[0] = value;
    if(value < _y_h_x_max_min[1]) _y_h_x_max_min[1] = value;

    map(&value, *(_y_h_x_max_min), *(_y_h_x_max_min + 1), (double)1.0, (double)0.0);

    qDebug() << "x: " << y_h << value << ": " << *(_y_h_x_max_min) << *(_y_h_x_max_min + 1);

    _anchor_user->setPos(value, _anchor_user->getPos().y());
}

void Tracker2DWindow::onGesticRegressorSVMPredictY(double y_h, double *)
{
    if(!_anchor_user->isVisible()){
        _anchor_user->setPos(0, 1);

        _timer->stop();
        _anchor->hide();
        _anchor_user->show();
    }

    double value = y_h;

    if(value > _y_h_y_max_min[0]) _y_h_y_max_min[0] = value;
    if(value < _y_h_y_max_min[1]) _y_h_y_max_min[1] = value;

    map(&value, *(_y_h_y_max_min), *(_y_h_y_max_min + 1), (double)1.0, (double)0.0);

    qDebug() << "y: " << y_h << value << ": " << *(_y_h_y_max_min) << *(_y_h_y_max_min + 1);

    _anchor_user->setPos(_anchor_user->getPos().x(), value);
}






AnchorWidget::AnchorWidget(QColor c, QWidget *parent) : QWidget(parent)
{
    _color = c;

    _anchor_size = 20;
    this->resize(640, 640);

    _current_pos = QPoint(0, height());
    update();
}


void AnchorWidget::setPos(double x_, double y_)
{

    if(x_ > 1) x_ = 1;
    if(x_ < 0) x_ = 0;

    if(y_ > 1) y_ = 1;
    if(y_ < 0) y_ = 0;

    double new_x = x_, new_y = y_;


    map(&new_x, 1, 0, width(), 0);
    map(&new_y, 1, 0, height(), 0);

    _current_pos.setX(new_x);
    _current_pos.setY(new_y);


    update();
}

void AnchorWidget::setColor(QColor c_)
{
    _color = c_;
}


QPointF AnchorWidget::getPos()
{
    double x_ = _current_pos.x(), y_ = _current_pos.y();

    map(&x_, width(), 0, 1, 0);
    map(&y_, height(), 0, 1, 0);

    return QPointF(x_, y_);
}


void AnchorWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    QPen pen(Qt::black, 4);
    painter.setPen(pen);

    painter.drawRect(2, 2, width()-4, height()-4);

    pen.setWidth(1);
    painter.setPen(pen);
    painter.setBrush(_color);

    painter.drawEllipse(_current_pos, _anchor_size, _anchor_size);
}

void AnchorWidget::map(double *v, double v_max, double v_min, double target_max, double target_min)
{
    *v = ((*v) - v_min) / (v_max - v_min) * (target_max - target_min) + target_min;
}
