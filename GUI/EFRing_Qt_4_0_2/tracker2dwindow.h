#ifndef TRACKER2DWINDOW_H
#define TRACKER2DWINDOW_H

#include <QObject>
#include <QWidget>
#include <QtWidgets>
#include <svm.h>

class AnchorWidget : public QWidget
{
    Q_OBJECT

public :
    AnchorWidget(QColor, QWidget *parent = 0);
    void setPos(double, double);
    void setColor(QColor);
    QPointF getPos();

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

private:
    void map(double*, double, double, double, double);

private:
    QColor _color;
    QPointF _current_pos;
    int _anchor_size;

};

class Tracker2DWindow : public QWidget
{
    Q_OBJECT
public:
    explicit Tracker2DWindow(QWidget *parent = nullptr);

protected:
    void resizeEvent(QResizeEvent* event) Q_DECL_OVERRIDE;
    void showEvent(QShowEvent *event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;

private:
    void setDefaultSVMParamX();
    void setDefaultSVMParamY();
    void stopRecording();
    void map(double*, double, double, double, double);

signals:
    void tracker2dWindowReadySignal();
    void tracker2dWindowStartRecordSignal();
    void tracker2dWindowSaveFileSignal(QList<double>);
    void tracker2dWindowCloseSignal();
    void tracker2dWindowStartTrainingSignal(QList<double>, QList<double>, svm_parameter, svm_parameter);
    void forceCalibrateSignal();
    void stopRecordingSignal();
    void tracker2dLoadModelXSignal();
    void tracker2dLoadModelYSignal();
    void tracker2dLoadDetectionModelSignal();

private slots:
    void onTimerTimeOut();
    void onLoadModelXActionTriggered();
    void onLoadModelYActionTriggered();
    void onLoadDetectionModelActionTriggered();

public slots:
    void onGesticDataRecorded();
    void onGesticRegressorSVMPredictX(double, double*);
    void onGesticRegressorSVMPredictY(double, double*);

private:
    AnchorWidget *_anchor = nullptr;
    AnchorWidget *_anchor_user = nullptr;
    QTimer *_timer;
    double _speed_x, _speed_y, dx_, dy_;

    double _last_pos_x, _last_pos_y;

    struct svm_parameter _param_x, _param_y;
    bool _is_recording;
    QList<double> _pos_x_list, _pos_y_list;
    double _y_h_x_max_min[2], _y_h_y_max_min[2];
};

#endif // TRACKER2DWINDOW_H
