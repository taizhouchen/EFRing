#include <stdio.h>
#include <windows.h>
#include <mygestic.h>
#include <stdbool.h>

int gestic_init(){

    GESTIC_INITIALIZED = false;

    /* Create GestIC-Instance */
    gestic = gestic_create();

    /* Bitmask later used for starting a stream with SD- and position-data */
    const int stream_flags = gestic_data_mask_cic | gestic_data_mask_sd;

    /* Initialize all variables and required resources of gestic */
    gestic_initialize(gestic);

    /* Aquire the pointers to the data-buffers */
    cic = gestic_get_cic(gestic, 0);
    sd = gestic_get_sd(gestic, 0);


    /* Try to open a connection to the device */
    if(gestic_open(gestic) < 0) {
        fprintf(stderr, "Could not open connection to device.\n");
        my_gestic_error = MY_GESTIC_NOT_DEVICE;
        return -1;
    }

    /* Try to reset the device to the default state:
     * - Automatic calibration enabled
     * - All frequencies allowed
     * - Approach detection disabled
     */
    if(
            gestic_set_auto_calibration(gestic, 0, 100) < 0 ||
            gestic_select_frequencies(gestic, gestic_freq1, 100) < 0 ||
            gestic_set_approach_detection(gestic, false, 100) < 0
       )
    {
        fprintf(stderr, "Could not reset device to default state.\n");
        my_gestic_error = MY_GESTIC_NOT_RESET_TO_DEFAULT;
        return -1;
    }

    /* Set output-mask to the bitmask defined above and stream all data */
    if(gestic_set_output_enable_mask(gestic, stream_flags, stream_flags,
                                     gestic_data_mask_all, 100) < 0)
    {
        fprintf(stderr, "Could not set output-mask for streaming.\n");
        my_gestic_error = MY_GESTIC_NOT_OUTPUT_MASKING;
        return -1;
    }

    my_gestic_error = MY_GESTIC_NO_ERROR;
    GESTIC_INITIALIZED = true;

    return 1;
}

int gestic_release(){

    /* Close connection to device */
    gestic_close(gestic);

    /* Release further resources that were used by gestic */
    gestic_cleanup(gestic);
    gestic_free(gestic);
    return 1;
}

float gestic_get_sd_from_channel( int channel ){
    if(my_gestic_error == MY_GESTIC_NO_ERROR){
        if(!gestic_data_stream_update(gestic, 0)){
            return sd->channel[channel];
        }
    }
    else return -1;
}



