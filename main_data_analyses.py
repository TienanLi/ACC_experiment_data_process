import os
from oscillation_functions import read_oscillation_data,draw_oscillation_statistics,draw_oscillation_statistics_2,\
    draw_oscillation_statistics_multiple_val

def oscillation_analyses():
    data=read_oscillation_data(os.getcwd() + '/platooned_data/02-25-2020/')
    # draw_oscillation_statistics(data,6,2,0,'Speed Level','Deceleration starting difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,3,0,'Speed Level','Deceleration ending difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,4,0,'Speed Level','Acceleration starting difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,5,0,'Speed Level','Acceleration ending difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,7,0,'Speed Level','Minimum speed difference (FV-LV) ($m$/$s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,8,0,'Speed Level','Ending speed difference (FV-LV) ($m$/$s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,9,0,'Speed Level','Acceleration rate difference (FV-LV) ($m$/$s^2$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,10,0,'Speed Level','Deceleration rate difference (FV-LV) ($m$/$s^2$)',stick_plot=True)


    draw_oscillation_statistics_2(data,6,2,0,18,'Speed Level','Deceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,3,0,18,'Speed Level','Deceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,4,0,18,'Speed Level','Acceleration starting difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,5,0,18,'Speed Level','Acceleration ending difference (s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,7,0,18,'Speed Level','Minimum speed difference (m per s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,8,0,18,'Speed Level','Ending speed difference (m per s)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,9,0,18,'Speed Level','Acceleration rate difference (m per s2)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,10,0,18,'Speed Level','Deceleration rate difference (m per s2)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,14,0,18,'Speed Level','FV acceleration rate (m per s2)',stick_plot=True)
    draw_oscillation_statistics_2(data,6,15,0,18,'Speed Level','FV deceleration rate (m per s2)',stick_plot=True)

    # draw_oscillation_statistics(data, 11, 2, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration starting difference (s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data,11, 3, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration ending difference (s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 4, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration starting difference (s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 5, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration ending difference (s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 7, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Minimum speed difference (m per s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 8, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Ending speed difference (m per s)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 9, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration rate difference (m per s2)',title='power mode',regression=True)
    # draw_oscillation_statistics(data, 11, 10, 1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration rate difference (m per s2)',title='power mode',regression=True)

    # draw_oscillation_statistics(data, 11, 2, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration starting difference (s)',title='speed range',regression=True)
    # draw_oscillation_statistics(data,11, 3, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration ending difference (s)',title='speed range',regression=True)
    # draw_oscillation_statistics(data, 11, 4, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration starting difference (s)',title='speed rangep',regression=True)
    # draw_oscillation_statistics(data, 11, 5, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration ending difference (s)',title='speed range',regression=True)
    # draw_oscillation_statistics(data, 11, 7, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Minimum speed difference (m per s)',title='speed range',regression=True)
    # draw_oscillation_statistics(data, 11, 8, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Ending speed difference (m per s)',title='speed range',regression=True)
    # draw_oscillation_statistics(data, 11, 9, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration rate difference (m per s2)',title='speed range',regression=True)
    # draw_oscillation_statistics(data, 11, 10, 0, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration rate difference (m per s2)',title='speed range',regression=True)

    # draw_oscillation_statistics(data, 11, 2, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration starting difference (s)',title='all',regression=True)
    # draw_oscillation_statistics(data,11, 3, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration ending difference (s)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 4, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration starting difference (s)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 5, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration ending difference (s)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 7, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Minimum speed difference (m per s)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 8, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Ending speed difference (m per s)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 9, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Acceleration rate difference (m per s2)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 10, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'Deceleration rate difference (m per s2)',title='all',regression=True)

    # draw_oscillation_statistics(data, 12, 9, -1, 'LV Acceleration rate (m per s2)',
    #                             'Acceleration rate difference (m per s2)',title='all',regression=True)
    # draw_oscillation_statistics(data, 13, 10, -1, 'LV Deceleration rate (m per s2)',
    #                             'Deceleration rate difference (m per s2)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 14, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'FV acceleration rate (m per s2)',title='all',regression=True)
    # draw_oscillation_statistics(data, 11, 15, -1, 'Oscillation magnitude of LV (m per s)',
    #                             'FV deceleration rate (m per s2)',title='all',regression=True)

    draw_oscillation_statistics_multiple_val(data, 6, [2,4,5], 0, 'Speed level',
                                'Delay ($s$)', y_limit = [0, 15])
    draw_oscillation_statistics_multiple_val(data, 6, [12,14], 0, 'Speed level',
                                'Acceleration rate ($m$/$s^2$)', y_limit = [0, 3], stick_plot=True)
    draw_oscillation_statistics_multiple_val(data, 6, [13,15], 0, 'Speed level',
                                'Deceleration rate ($m$/$s^2$)', y_limit = [0, 3] ,stick_plot=True)

if __name__ == '__main__':
    oscillation_analyses()
