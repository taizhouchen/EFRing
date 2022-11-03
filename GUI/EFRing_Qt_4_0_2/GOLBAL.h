#ifndef GOLBAL_H
#define GOLBAL_H

#include <QtCore>

typedef enum {
    SD = 1,
    CIC = 2
} GESTIC_DATA_TYPE;

typedef enum {
    UNDEFINE = -2,
    CONTINUOUD_CIRCULAR_SLIDER = 'c',
    CONTINUOUS_LINEAR_SLIDER = 'l',
    NONE = -1,
    TAP = 0,
    DOUBLE_TAP = 1,
    SWIPE_LEFT = 2,
    SWIPE_RIGHT = 3,
    SWIPE_UP = 4,
    SWIPE_DOWN = 5,
    CHECK = 6,
    CIRCLE_CLOCKWISE = 7,
    CIRCLE_COUNTERCLOCKWISE = 8,
} LABELS;

struct label_img_off{
    QString TAP = "tap_off.png";
    QString DOUBLE_TAPG = "doubletap_off.png";
    QString SWIPE_LEFT = "swipe_left_off.png";
    QString SWIPE_RIGHT = "swipe_right_off.png";
    QString SWIPE_UP = "swipe_up_off.png";
    QString SWIPE_DOWN = "swipe_down_off.png";
    QString CHECK = "check_off.png";
    QString CIRCLE_CLOCKWISE = "circle_clockwise_off.png";
    QString CIRCLE_COUNTERCLOCKWISE = "circle_counterclockwise_off.png";
};

struct label_img_on{
    QString TAP = "tap_on.png";
    QString DOUBLE_TAPG = "doubletap_on.png";
    QString SWIPE_LEFT = "swipe_left_on.png";
    QString SWIPE_RIGHT = "swipe_right_on.png";
    QString SWIPE_UP = "swipe_up_on.png";
    QString SWIPE_DOWN = "swipe_down_on.png";
    QString CHECK = "check_on.png";
    QString CIRCLE_CLOCKWISE = "circle_clockwise_on.png";
    QString CIRCLE_COUNTERCLOCKWISE = "circle_counterclockwise_on.png";
};

static QString LABEL_IMG_PATH = "../preselect_gestures/";
static label_img_off LABEL_IMG_OFF;
static label_img_on LABEL_IMG_ON;

static QString DATA_FILE_PREFIXES = "gestic_data_";
static QString RESULT_FILE_PREFIXES = "gestic_result_";

// recording duration in ms
static int MICROGESTURE_LENGTH_ = 1000;
static int LINEARGESTURE_LENGTH_MAXIMUM_ = 10000;
static int DETECTOR_RECORD_LENGTH_ = 10000;

extern int COUNTER_;

extern LABELS CURRENT_LABEL_;

extern QDateTime TIME_STEMP_FOR_RECORD_;

#endif // GOLBAL_H
