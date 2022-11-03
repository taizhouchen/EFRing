#ifndef AUDIORECORDINGTHREAD_H
#define AUDIORECORDINGTHREAD_H

#include <QObject>
#include <QAudioRecorder>
#include <QDebug>
#include <QThread>
#include <QTime>


class AudioRecordingThread : public QThread
{
    Q_OBJECT
public:
    explicit AudioRecordingThread(QObject *parent = nullptr);
    void run();

    QStringList getInputDevices();
    QList<int> getSupportedSampleRates();
    QAudioRecorder* getRecorder(){ return m_audioRecorder; }

    void setInputDevice(QString d){ _audiodevice = d;}
    void setSampleRate(int sr) { _sample_rate = sr; }
    void setBitRate(int br) { _bitrate = br; }
    void setFilePath(QString p){ _path_to_save = p; }

    // recording length in ms
    void setRecordLength(int l) {
        _record_length = l;
    }

    void startRecord();
    bool keepTheFile();
    void removeFile(QString);
    void renameFile(QString, QString);

    QAudioRecorder::Status getRecorderStatue();
    void enable();
    void disable();

private:
    void toggleRecord();

private slots:
    void onRecorderStateChanged(QMediaRecorder::State);

signals:
    void audioRecordingThreadStartRecording();

private:
    QAudioRecorder *m_audioRecorder = nullptr;
    QString _codec = "audio/pcm";
    QString _audiodevice = NULL;
    QString _path_to_save;
    int _sample_rate = 44100;
    int _bitrate = 88200;

    int _record_length;
    QTime _record_till;
    bool _is_recording = false;
    bool _enabled;

    QString _current_file = NULL;

    bool _keep_the_file = false;

signals:

};

#endif // AUDIORECORDINGTHREAD_H
