#include "datasmoothor.h"

ExpMovAvg::ExpMovAvg(double decay, QObject *parent) : QObject(parent)
{
    _decay = decay;
    _is_first = true;
}

void ExpMovAvg::update(double *d)
{
    if(_is_first){
        _last_sample = *d;
        _is_first = false;
    }else{
        *d = _decay * _last_sample + (1 - _decay) * *d;
        _last_sample = *d;
    }
}

void ExpMovAvg::reset()
{
    _is_first = true;
}





MovAvg::MovAvg(int window_size, QObject *parent) : QObject(parent)
{
    _window_size = window_size;
    _data_queue = new double[_window_size];
    _queue_index = 0;
}

void MovAvg::update(double *d)
{
    if(_queue_index < _window_size){
        _data_queue[_queue_index++] = *d;
        return;
    }else if(_queue_index == _window_size){
        // pop the first element and shift the rest of elements to left
        for(int i = 0; i < _window_size - 1; i++){
            _data_queue[i] = _data_queue[i+1];
        }
        // push new elemeny to the end
        _data_queue[_window_size - 1] = *d;
    }

    double sum = 0;
    for(int i = 0; i < _window_size; i++){
        sum += _data_queue[i];
    }

    *d = sum / _window_size;
}

void MovAvg::reset()
{
    _queue_index = 0;
}

void MovAvg::release()
{
    free(_data_queue);
}




KF::KF(QObject *parent)
{
    _n = 3; // Number of states
    _m = 1; // Number of measurements

    _dt = 1.0/190; // Time step

    _F = Eigen::MatrixXd(_n, _n); // System dynamics matrix
    _H = Eigen::MatrixXd(_m, _n); // Output matrix
    _Q = Eigen::MatrixXd(_n, _n); // Process noise covariance
    _R = Eigen::MatrixXd(_m, _m); // Measurement noise covariance
    _P = Eigen::MatrixXd(_n, _n); // Estimate error covariance

    // Discrete LTI projectile motion, measuring position only
    _F << 1, _dt, 0, 0, 1, _dt, 0, 0, 1;
    _H << 1, 0, 0;

    // Reasonable covariance matrices
    _Q << .05, .05, .0, .05, .05, .0, .0, .0, .0;
    _R << 1;
    _P << .1, .1, .1, .1, 10000, 10, .1, 10, 100;

//    _F << 1, _dt, 0, 1;
//    _H << 1, 0;

//    _Q << 0.1, 0, 0, 0.1;
//    _R << 3;
//    _P << .1, 0, 0, .1;

    reset();
}

void KF::update(double *d)
{
    Eigen::VectorXd y(_m);
    y << *d;

    _kf.update(y);

    *d = _kf.state().transpose()[0];
}

void KF::reset()
{
    // Construct the filter
    _kf = KalmanFilter(_dt, _F, _H, _Q, _R, _P);

    // Best guess of initial states
//    Eigen::VectorXd x0(_n);
//    double t = 0;
//    x0 << 0, 0, -9.8;
//    _kf.init(t, x0);
    _kf.init();
}

