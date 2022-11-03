#ifndef TRIALMODEWINDOW_H
#define TRIALMODEWINDOW_H

#include <QWidget>
#include <QtWidgets>
#include <GOLBAL.h>

class TrialModeWindow : public QWidget
{
    Q_OBJECT
public:
    explicit TrialModeWindow(QWidget *parent = nullptr);
    void setLabel(int, LABELS);

protected:
    void showEvent(QShowEvent *event) Q_DECL_OVERRIDE;
    void resizeEvent(QResizeEvent* event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;


signals:
    void startRecordSignal();
    void saveSignal();

private:
    QLabel *_image_container;
    QPixmap _img;

};

#endif // TRIALMODEWINDOW_H
