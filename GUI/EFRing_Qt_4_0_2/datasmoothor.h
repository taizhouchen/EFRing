#ifndef DATASMOOTHOR_H
#define DATASMOOTHOR_H

#include <QObject>
#include <QWidget>
#include <QDebug>
#include <kalman.hpp>
#include <Eigen/Dense>

class ExpMovAvg : public QObject
{
    Q_OBJECT
public:
    explicit ExpMovAvg(double, QObject *parent = nullptr);
    void update(double*);
    void reset();

private:
    double _last_sample, _decay;
    bool _is_first;

};

class MovAvg : public QObject
{
    Q_OBJECT
public:
    explicit MovAvg(int, QObject *parent = nullptr);
    void update(double*);
    void reset();
    void release();

private:
    int _window_size, _queue_index;
    double *_data_queue;
    bool _is_forst;
};

class KF : public QObject{
    Q_OBJECT
public:
    explicit KF(QObject *parent = nullptr);
    void update(double*);
    void reset();

private:
    KalmanFilter _kf;
    int _n,     // number of states
        _m;     // number of measurement
    double _dt;
    Eigen::MatrixXd _F, _H, _Q, _R, _P;
};

#endif // DATASMOOTHOR_H
