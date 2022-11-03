QT       += core gui multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    GOLBAL.cpp \
    audiorecordingthread.cpp \
    datasmoothor.cpp \
    detectorwindow.cpp \
    gesticregressorsvm.cpp \
    gesticstreamingthread.cpp \
    linearslider.cpp \
    linearslidertrial.cpp \
    main.cpp \
    mainwindow.cpp \
    mygestic.c \
    qcustomplot.cpp \
    splashscreen.cpp \
    svm.cpp \
    angularsliderwindow.cpp \
    tracker2dwindow.cpp \
    trialmodewindow.cpp \
    kalman.cpp

HEADERS += \
    GOLBAL.h \
    audiorecordingthread.h \
    datasmoothor.h \
    detectorwindow.h \
    gesticregressorsvm.h \
    gesticstreamingthread.h \
    linearslider.h \
    linearslidertrial.h \
    mainwindow.h \
    mygestic.h \
    qcustomplot.h \
    splashscreen.h \
    svm.h \
    angularsliderwindow.h \
    tracker2dwindow.h \
    trialmodewindow.h \
    kalman.hpp

FORMS += \
    linearslider.ui \
    mainwindow.ui \
    splash_screen.ui


unix|win32: LIBS += -L'C:/Program Files (x86)/Microchip/GestIC SDK 1.2.0/api/lib/' -lgestic

INCLUDEPATH += 'C:/Program Files (x86)/Microchip/GestIC SDK 1.2.0/api/include' \
                Eigen/

DEPENDPATH += 'C:/Program Files (x86)/Microchip/GestIC SDK 1.2.0/api/include'
