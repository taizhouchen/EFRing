#ifndef MYGESTIC_H
#define MYGESTIC_H

#include <gestic_api.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MY_GESTIC_NO_ERROR = 0,
    MY_GESTIC_NOT_DEVICE = -1,
    MY_GESTIC_NOT_RESET_TO_DEFAULT = -2,
    MY_GESTIC_NOT_OUTPUT_MASKING = -3
} my_gestic_error;


gestic_t *gestic;
const gestic_signal_t * sd;
const gestic_signal_t * cic;
bool GESTIC_INITIALIZED;

int gestic_init();
int gestic_release();
float gestic_get_sd_from_channel( int channel );


#ifdef __cplusplus
}
#endif

#endif // MYGESTIC_H
