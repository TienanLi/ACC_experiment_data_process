from oscillation_functions import read_oscillation_data,draw_oscillation_statistics,draw_oscillation_statistics_2

def oscillation_analyses():
    data=read_oscillation_data()
    draw_oscillation_statistics(data,6,2,0,'Oscillation starting speed of LV (kph)','Deceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics(data,6,3,0,'Oscillation starting speed of LV (kph)','Deceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics(data,6,4,0,'Oscillation starting speed of LV (kph)','Acceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics(data,6,5,0,'Oscillation starting speed of LV (kph)','Acceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics(data,6,7,0,'Oscillation starting speed of LV (kph)','Minimum speed difference (kph)',stick_plot=True)
    draw_oscillation_statistics(data,6,8,0,'Oscillation starting speed of LV (kph)','Ending speed difference (kph)',stick_plot=True)
    draw_oscillation_statistics(data,6,9,0,'Oscillation starting speed of LV (kph)','Acceleration rate difference (m per s2)',stick_plot=True)
    draw_oscillation_statistics(data,6,10,0,'Oscillation starting speed of LV (kph)','Deceleration rate difference (m per s2)',stick_plot=True)

    draw_oscillation_statistics_2(data,6,2,0,1,'Oscillation starting speed of LV (kph)','Deceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,3,0,1,'Oscillation starting speed of LV (kph)','Deceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,4,0,1,'Oscillation starting speed of LV (kph)','Acceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,5,0,1,'Oscillation starting speed of LV (kph)','Acceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,7,0,1,'Oscillation starting speed of LV (kph)','Minimum speed difference (kph)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,8,0,1,'Oscillation starting speed of LV (kph)','Ending speed difference (kph)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,9,0,1,'Oscillation starting speed of LV (kph)','Acceleration rate difference (m per s2)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,10,0,1,'Oscillation starting speed of LV (kph)','Deceleration rate difference (m per s2)',stick_plot=True)


    draw_oscillation_statistics(data, 11, 2, 1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration starting difference (s)',title='power mode',regression=True)
    draw_oscillation_statistics(data,11, 3, 1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration ending difference (s)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 4, 1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration starting difference (s)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 5, 1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration ending difference (s)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 7, 1, 'Oscillation magnitude of LV (kph)',
                                'Minimum speed difference (kph)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 8, 1, 'Oscillation magnitude of LV (kph)',
                                'Ending speed difference (kph)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 9, 1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration rate difference (m per s2)',title='power mode',regression=True)
    draw_oscillation_statistics(data, 11, 10, 1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration rate difference (m per s2)',title='power mode',regression=True)

    draw_oscillation_statistics(data, 11, 2, 0, 'Oscillation magnitude of LV (kph)',
                                'Deceleration starting difference (s)',title='speed range',regression=True)
    draw_oscillation_statistics(data,11, 3, 0, 'Oscillation magnitude of LV (kph)',
                                'Deceleration ending difference (s)',title='speed range',regression=True)
    draw_oscillation_statistics(data, 11, 4, 0, 'Oscillation magnitude of LV (kph)',
                                'Acceleration starting difference (s)',title='speed rangep',regression=True)
    draw_oscillation_statistics(data, 11, 5, 0, 'Oscillation magnitude of LV (kph)',
                                'Acceleration ending difference (s)',title='speed range',regression=True)
    draw_oscillation_statistics(data, 11, 7, 0, 'Oscillation magnitude of LV (kph)',
                                'Minimum speed difference (kph)',title='speed range',regression=True)
    draw_oscillation_statistics(data, 11, 8, 0, 'Oscillation magnitude of LV (kph)',
                                'Ending speed difference (kph)',title='speed range',regression=True)
    draw_oscillation_statistics(data, 11, 9, 0, 'Oscillation magnitude of LV (kph)',
                                'Acceleration rate difference (m per s2)',title='speed range',regression=True)
    draw_oscillation_statistics(data, 11, 10, 0, 'Oscillation magnitude of LV (kph)',
                                'Deceleration rate difference (m per s2)',title='speed range',regression=True)

    draw_oscillation_statistics(data, 11, 2, -1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration starting difference (s)',title='all',regression=True)
    draw_oscillation_statistics(data,11, 3, -1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration ending difference (s)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 4, -1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration starting difference (s)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 5, -1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration ending difference (s)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 7, -1, 'Oscillation magnitude of LV (kph)',
                                'Minimum speed difference (kph)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 8, -1, 'Oscillation magnitude of LV (kph)',
                                'Ending speed difference (kph)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 9, -1, 'Oscillation magnitude of LV (kph)',
                                'Acceleration rate difference (m per s2)',title='all',regression=True)
    draw_oscillation_statistics(data, 11, 10, -1, 'Oscillation magnitude of LV (kph)',
                                'Deceleration rate difference (m per s2)',title='all',regression=True)

    draw_oscillation_statistics(data, 12, 9, -1, 'LV Acceleration rate (m per s2)',
                                'Acceleration rate difference (m per s2)',title='all',regression=True)
    draw_oscillation_statistics(data, 13, 10, -1, 'LV Deceleration rate (m per s2)',
                                'Deceleration rate difference (m per s2)',title='all',regression=True)


if __name__ == '__main__':
    oscillation_analyses()
