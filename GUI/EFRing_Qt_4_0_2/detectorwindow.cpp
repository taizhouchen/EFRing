#include "detectorwindow.h"

DetectorWindow::DetectorWindow(QWidget *parent) : QWidget(parent)
{
    _label_offring = new QLabel(this);
    _label_offring->setText("<html><head/><body><p><span style=\" font-size:128pt; font-weight:600; color:#2ecc71;\">OFF RING</span></p></body></html>");
    _label_offring->setAlignment(Qt::AlignCenter);

    _label_none = new QLabel(this);
    _label_none->setText("<html><head/><body><p><span style=\" font-size:128pt; font-weight:600; color:#34495e;\">NO CONTACT</span></p></body></html>");
    _label_none->setAlignment(Qt::AlignCenter);

    _label_onring = new QLabel(this);
    _label_onring->setText("<html><head/><body><p><span style=\" font-size:128pt; font-weight:600; color:#3498db;\">ON RING</span></p></body></html>");
    _label_onring->setAlignment(Qt::AlignCenter);

    _textInstruction = new QTextBrowser(this);
    _textInstruction->viewport()->setAutoFillBackground(false);
    _textInstruction->setText("Press [0] to record negative sample \nPress [1] to record positive sample \nPress [T] to start training \nPress [s] to save the model \nPress [c] to force calibrate (please move your thumb to the distal phalanx first before doing this)");

    QMenuBar *menu_bar = new QMenuBar(this);
    QAction *tool_action = new QAction("Load Model", this);
    QAction *train_from_file_action = new QAction("Train From File", this);
    QMenu *tool_menu = menu_bar->addMenu("Tools");

    tool_menu->addAction(tool_action);
    tool_menu->addAction(train_from_file_action);

    connect(tool_action, &QAction::triggered, this, &DetectorWindow::onLoadModelActionTriggered);
    connect(train_from_file_action, &QAction::triggered, this, &DetectorWindow::onTrainFromFile);
}

void DetectorWindow::setFileName(QString file_name)
{
    _file_name = file_name;
}

void DetectorWindow::showEvent(QShowEvent *event)
{
    setWindowTitle(tr("Detector"));
    resize(1280, 720);

    _is_recording = false;

    _label_offring->move(width()/2 - _label_offring->width()/2, height()/2 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()/2 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()/2 - _label_onring->height()/2);

    _label_offring->hide();
    _label_none->hide();
    _label_onring->hide();

    _svm_point_none.clear();
    _svm_point_offring.clear();
    _svm_point_onring.clear();

    _calibration_base = 0;
    _calibration_base_list.clear();
    _is_calibrating = false;

    setDefaultSVMParam();


    emit detectorReadySignal();
}

void DetectorWindow::resizeEvent(QResizeEvent *event)
{
    _label_offring->move(width()/2 - _label_offring->width()/2, height()/2 - _label_offring->height()/2);
    _label_none->move(width()/2 - _label_none->width()/2, height()/2 - _label_none->height()/2);
    _label_onring->move(width()/2 - _label_onring->width()/2, height()/2 - _label_onring->height()/2);

    _textInstruction->setGeometry(0, 0, width(), height());
    _textInstruction->move(10, height()-100);
    _textInstruction->setFrameStyle(QFrame::NoFrame);
}

void DetectorWindow::closeEvent(QCloseEvent *event)
{
    free(_param.weight_label);
    free(_param.weight);

    emit detectorWindowCloseSignal();
}

void DetectorWindow::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_0:
            if(!_is_recording){
                _label_list.clear();
                _current_training_label = DetectorLabel::NO_CONTACT;
                emit detectorStartRecordSignal(false);
            }
            break;
        case Qt::Key_1:
            if(!_is_recording){
                _label_list.clear();
                _current_training_label = DetectorLabel::OFF_RING;
                emit detectorStartRecordSignal(true);
            }
            break;
        case Qt::Key_2:
            if(!_is_recording){
                _label_list.clear();
                _current_training_label = DetectorLabel::ON_RING;
                emit detectorStartRecordSignal(false);
            }
            break;

        case Qt::Key_T:
            if(!_is_recording){
                if(!(_svm_point_none.size() == 0 && _svm_point_offring.size() == 0 && _svm_point_onring.size() == 0)){
                    setDefaultSVMParam();
                    QList<SVMPoint> _svm_point_all = _svm_point_none + _svm_point_offring + _svm_point_onring;
                    emit detectorStartTrainingSignal(_svm_point_all, _param);
                }else{
                    QMessageBox msgBox;
                    msgBox.setIcon(QMessageBox::Warning);
                    msgBox.setText("No enough data to train!");
                    msgBox.setWindowTitle("Message");
                    msgBox.exec();
                }
            }
            break;
        case Qt::Key_S:
            if(!_is_recording) saveToFile(_file_name);
            break;
        case Qt::Key_C:
            _calibration_base_list.clear();
            _is_calibrating = true;
            emit forceCalibrateSignal();

            break;

        case Qt::Key_Escape:
            this->close();
            break;
    }
}

void DetectorWindow::setDefaultSVMParam()
{
    // default values
    _param.svm_type = C_SVC;
    _param.kernel_type = RBF;
    _param.degree = 3;
    _param.gamma = 4;	// 1/num_features
    _param.coef0 = 0;
    _param.nu = 0.5;
    _param.cache_size = 100;
    _param.C = 4;
    _param.eps = 1e-3;
    _param.p = 0.1;
    _param.shrinking = 1;
    _param.probability = 0;
    _param.nr_weight = 0;
    _param.weight_label = NULL;
    _param.weight = NULL;
}

void DetectorWindow::saveToFile(QString file_name)
{
    if(!(_svm_point_none.size() == 0 && _svm_point_offring.size() == 0 && _svm_point_onring.size() == 0)){
        if(!file_name.endsWith("csv")){
            file_name += ".csv";
        }

        QFile file(file_name);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)){
            qDebug() << "Cannot open file";
            return;
        }

        QTextStream out(&file);

        QList<SVMPoint> _svm_point_all = _svm_point_none + _svm_point_offring + _svm_point_onring;

        // write headder
        out << '\t' << ','
            << TIME_STEMP_FOR_RECORD_.toString("yyyyMMddHHmmss") << ','
            << QString::number(_svm_point_all.size()) << '\n';

        if(_svm_point_all.size() != 0){
            for(QList<SVMPoint>::Iterator p = _svm_point_all.begin(); p != _svm_point_all.end(); p++){
                out << p->sd_c0 << ','
                    << p->sd_c1 << ','
                    << p->sd_c2 << ','
                    << p->sd_c3 << ','
                    << p->sd_c4 << ','
                    << p->cic_c0 << ','
                    << p->cic_c1 << ','
                    << p->cic_c2 << ','
                    << p->cic_c3 << ','
                    << p->cic_c4 << ','
                    << p->y << '\n';
            }
        }

        file.close();

        qDebug() << QString("%1 lines of data were saved to %2").arg(_svm_point_all.size()).arg(file_name);
    }else{
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("No enough data to save!");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }
}

double DetectorWindow::findMean(QList<double> a)
{
    double sum = 0;
    for(QList<double>::Iterator p = a.begin(); p != a.end(); p++) sum += *p;

    return sum/a.size();
}

void DetectorWindow::onTrainFromFile()
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
            emit detectorStartTrainingSignal(_svm_point_all, _param);
        }
    }
}

void DetectorWindow::onGesticDataRecorded()
{
    _is_recording = true;

    _label_list.append(_current_training_label);
}


void DetectorWindow::onGesticDataStopRecording()
{
    _is_recording = false;

    emit detectorGetSVMPointSignal(_label_list);
}

void DetectorWindow::onReturnSVMPoint(QList<SVMPoint> svm_point_)
{
    if(_current_training_label == DetectorLabel::NO_CONTACT)
        _svm_point_none += svm_point_;
    if(_current_training_label == DetectorLabel::OFF_RING)
        _svm_point_offring += svm_point_;
    if(_current_training_label == DetectorLabel::ON_RING)
        _svm_point_onring += svm_point_;

}

void DetectorWindow::onGesticDetectorSVMPredict(double y_h, double *)
{
//    _kf->update(&y_h);

//    qDebug() << y_h << y_h - _calibration_base;

//    if(_is_calibrating){
//        if(_calibration_base_list.size() < 100){
//            _calibration_base_list.append(y_h);
//        }else{
//            qSort(_calibration_base_list.begin(), _calibration_base_list.end());

//            _calibration_base = _calibration_base_list[int(_calibration_base_list.size() / 2)]; // get the medium

//            _calibration_base -= 0.1;
//            _is_calibrating = false;
//        }
//    }

//    if(y_h - _calibration_base < 0){
//            _label_offring->hide();
//            _label_onring->hide();
//            _label_none->show();
//    }else if(y_h - _calibration_base > 1){
//            _label_offring->hide();
//            _label_none->hide();
//            _label_onring->show();
//    }else{
//            _label_onring->hide();
//            _label_none->hide();
//            _label_offring->show();
//    }

    switch (int(y_h)) {
        case DetectorLabel::NO_CONTACT:
            _label_offring->hide();
            _label_onring->hide();
            _label_none->show();
            break;
        case DetectorLabel::OFF_RING:
            _label_onring->hide();
            _label_none->hide();
            _label_offring->show();
            break;
        case DetectorLabel::ON_RING:
            _label_offring->hide();
            _label_none->hide();
            _label_onring->show();
            break;
    }
}

void DetectorWindow::onLoadModelActionTriggered()
{
    emit detectorLoadModelSignal();
}

double DetectorWindow::labelToRegressorValue(double label)
{
    switch(int(label)){
        case DetectorLabel::NO_CONTACT:
            return 0;
            break;
        case DetectorLabel::OFF_RING:
            return 0.5;
            break;
        case DetectorLabel::ON_RING:
            return 1.0;
            break;
    }
}

