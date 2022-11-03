#include "trialmodewindow.h"

TrialModeWindow::TrialModeWindow(QWidget *parent) : QWidget(parent)
{
//    this->setWindowFlags(Qt::FramelessWindowHint);
    this->setStyleSheet("background-color:white;");

    _image_container = new QLabel(this);
    _image_container->setAlignment(Qt::AlignCenter);


    _image_container->setScaledContents(true);
}

void TrialModeWindow::showEvent(QShowEvent *event)
{
    setWindowTitle(tr("Trial Mode"));
    resize(1280, 720);

    this->move(QApplication::desktop()->screen()->rect().center() - this->rect().center());
    _image_container->setGeometry(width()/2-480, height()/2-270, 960, 540);

}

void TrialModeWindow::resizeEvent(QResizeEvent *event)
{
    _image_container->setGeometry(width()/2-480, height()/2-270, 960, 540);
}

void TrialModeWindow::closeEvent(QCloseEvent *event)
{

}

void TrialModeWindow::keyPressEvent(QKeyEvent *event)
{
    switch(event->key()){
        case Qt::Key_Escape:
            this->close();
            break;
        case Qt::Key_Space:
            _img.load(LABEL_IMG_PATH + LABEL_IMG_OFF.SWIPE_LEFT);
            update();
            break;
        case Qt::Key_F1:
            this->isFullScreen() ? this->showNormal() : this->showFullScreen();
            break;
        case Qt::Key_R:
            emit startRecordSignal();
            break;
        case Qt::Key_S:
            emit saveSignal();
            break;
    }
}

void TrialModeWindow::setLabel(int mode_, LABELS label_)
{
    if(mode_ != 0 && mode_ != 1) return;

    if(mode_ == 0){

        switch (label_) {
            case LABELS::TAP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.TAP));
                break;
            case LABELS::DOUBLE_TAP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.DOUBLE_TAPG));
                break;
            case LABELS::SWIPE_LEFT:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.SWIPE_LEFT));
                break;
            case LABELS::SWIPE_RIGHT:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.SWIPE_RIGHT));
                break;
            case LABELS::SWIPE_UP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.SWIPE_UP));
                break;
            case LABELS::SWIPE_DOWN:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.SWIPE_DOWN));
                break;
            case LABELS::CHECK:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.CHECK));
                break;
            case LABELS::CIRCLE_CLOCKWISE:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.CIRCLE_CLOCKWISE));
                break;
            case LABELS::CIRCLE_COUNTERCLOCKWISE:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_OFF.CIRCLE_COUNTERCLOCKWISE));
                break;
            default:
                return;
                break;
        }
    }

    if(mode_ == 1){
        switch (label_) {
            case LABELS::TAP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.TAP));
                break;
            case LABELS::DOUBLE_TAP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.DOUBLE_TAPG));
                break;
            case LABELS::SWIPE_LEFT:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.SWIPE_LEFT));
                break;
            case LABELS::SWIPE_RIGHT:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.SWIPE_RIGHT));
                break;
            case LABELS::SWIPE_UP:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.SWIPE_UP));
                break;
            case LABELS::SWIPE_DOWN:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.SWIPE_DOWN));
                break;
            case LABELS::CHECK:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.CHECK));
                break;
            case LABELS::CIRCLE_CLOCKWISE:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.CIRCLE_CLOCKWISE));
                break;
            case LABELS::CIRCLE_COUNTERCLOCKWISE:
                _image_container->setPixmap(QPixmap(LABEL_IMG_PATH + LABEL_IMG_ON.CIRCLE_COUNTERCLOCKWISE));
                break;
            default:
                return;
                break;
        }
    }

    update();

}


