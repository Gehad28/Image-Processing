QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    dialog.cpp \
    dialog2.cpp \
    dialog_canny.cpp \
    dialog_edgdetection.cpp \
    dialogglobalthreshold.cpp \
    dialogminmax.cpp \
    dialognois.cpp \
    dialognoise.cpp \
    filter.cpp \
    frequency.cpp \
    histogram.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    dialog.h \
    dialog2.h \
    dialog_canny.h \
    dialog_edgdetection.h \
    dialogglobalthreshold.h \
    dialogminmax.h \
    dialognois.h \
    dialognoise.h \
    filter.h \
    frequency.h \
    histogram.h \
    mainwindow.h

FORMS += \
    dialog.ui \
    dialog2.ui \
    dialog_canny.ui \
    dialog_edgdetection.ui \
    dialogglobalthreshold.ui \
    dialogminmax.ui \
    dialognois.ui \
    dialognoise.ui \
    mainwindow.ui

INCLUDEPATH += C:/opencv/release/install/include

LIBS += C:/opencv/release/bin/libopencv_core470.dll
LIBS += C:/opencv/release/bin/libopencv_highgui470.dll
LIBS += C:/opencv/release/bin/libopencv_imgcodecs470.dll
LIBS += C:/opencv/release/bin/libopencv_imgproc470.dll
LIBS += C:/opencv/release/bin/libopencv_calib3d470.dll

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
