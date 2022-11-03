#include "audiorecordingthread.h"
#include <QAudioRecorder>
#include <QMediaRecorder>
#include <QUrl>
#include <QDir>
#include "GOLBAL.h"
#include <QMessageBox>

AudioRecordingThread::AudioRecordingThread(QObject *parent) : QThread(parent)
{
    enable();
    m_audioRecorder = new QAudioRecorder(this);
//    _record_length = MICROGESTURE_LENGTH_;

    QAudioEncoderSettings settings;
    settings = m_audioRecorder->audioSettings();

    connect(m_audioRecorder, &QAudioRecorder::stateChanged, this, &AudioRecordingThread::onRecorderStateChanged);
    qDebug() << settings.bitRate() << ", " << settings.codec();
}

void AudioRecordingThread::run(){
    while(_enabled){
        while(_is_recording){
//            qDebug() << m_audioRecorder->duration();
            if(m_audioRecorder->duration() > _record_length){
                toggleRecord();
            }
        }
    }
}

QStringList AudioRecordingThread::getInputDevices()
{
    return m_audioRecorder->audioInputs();
}

QList<int> AudioRecordingThread::getSupportedSampleRates()
{
    return m_audioRecorder->supportedAudioSampleRates();
}

void AudioRecordingThread::startRecord()
{
    if(getRecorderStatue() != QMediaRecorder::UnloadedStatus){
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Recorder is no ready");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
    }else{
        toggleRecord();
    }
}

bool AudioRecordingThread::keepTheFile()
{
    if(getRecorderStatue() != QMediaRecorder::UnloadedStatus){
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Recorder is no ready for saving file. Try again later.");
        msgBox.setWindowTitle("Message");
        msgBox.exec();
        return false;
    }else{
        QString new_file = _current_file;
        renameFile(_current_file, new_file.replace("_temp", ""));
        return true;
    }
}

void AudioRecordingThread::removeFile(QString path_to_file)
{
    QFile file(path_to_file);
    if(file.exists()){
        file.remove();
//        qDebug() << "file " << path_to_file << " was removed";

    }
}

void AudioRecordingThread::renameFile(QString a, QString b)
{
    if(QFile::exists(a)){
        QFile::rename(a, b);
        qDebug() << QString("file %1 was saved").arg(b);
    }
}

QMediaRecorder::Status AudioRecordingThread::getRecorderStatue()
{
    return m_audioRecorder->status();
}

void AudioRecordingThread::enable()
{
    _enabled = true;
}

void AudioRecordingThread::disable()
{
    _enabled = false;
}

void AudioRecordingThread::toggleRecord()
{
    if (m_audioRecorder->state() == QMediaRecorder::StoppedState) {
        m_audioRecorder->setAudioInput(_audiodevice);

        QAudioEncoderSettings settings;
        settings.setCodec(_codec);
        settings.setSampleRate(_sample_rate);
        settings.setBitRate(_bitrate);
        settings.setChannelCount(1);
        settings.setQuality(QMultimedia::HighQuality);

        m_audioRecorder->setEncodingSettings(settings);

        //delete the last temp file
        removeFile(_current_file);

        if(_path_to_save != NULL){
            QDir dir = QDir(_path_to_save);
            if(!dir.exists()){
                dir.mkdir(".");
            }

            QString filebase = TIME_STEMP_FOR_RECORD_.toString("yyyyMMddHHmmss") \
                    + "_" + QString::number(COUNTER_) \
                    + "_" + QString::number(CURRENT_LABEL_) + "_temp.wav";

            _current_file = QDir::cleanPath(
                        QDir::currentPath() +
                        QDir::separator() +
                        dir.filePath(filebase));

            bool result_ = m_audioRecorder->setOutputLocation(QUrl::fromLocalFile(_current_file));

//            qDebug() << "Setting new file to: " << _current_file << result_;
        }

        m_audioRecorder->record();
    }
    else {
        m_audioRecorder->stop();
        _keep_the_file = false;
    }
}

void AudioRecordingThread::onRecorderStateChanged(QMediaRecorder::State new_state)
{
    if(new_state == QMediaRecorder::RecordingState){
        qDebug() << "Audio Record started!";
        _is_recording = true;
        emit audioRecordingThreadStartRecording();
    }

    if(new_state == QMediaRecorder::StoppedState){
        _is_recording = false;

        qDebug() << "Audio record finished!";
//        qDebug() << m_audioRecorder->outputLocation();

    }
}



