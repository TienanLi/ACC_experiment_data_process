import os
import numpy as np
import matplotlib.pyplot as plt
from oscillation_functions import read_oscillation_data,draw_oscillation_statistics,draw_oscillation_statistics_2,\
    draw_oscillation_statistics_multiple_val,draw_oscillation_statistics_2_and_more

def oscillation_analyses():
    data=read_oscillation_data(os.getcwd() + '/platooned_data/02-25-2020/')
    # for note in ['low', 'middle', 'high']:
    #     for note_2 in ['headway - 1', 'headway - 3']:
    #         print(note)
    #         print(note_2)
    #         print(np.mean([d[11] for d in data if d[18] == note and d[21] == note_2]))  # magnitude
    #         print(np.std([d[11] for d in data if d[18] == note and d[21] == note_2]))  # magnitude
    #         print(np.mean([d[13] for d in data if d[18] == note and d[21] == note_2]))#Dec
    #         print(np.std([d[13] for d in data if d[18] == note and d[21] == note_2]))  # Dec
    #         print(np.mean([d[12] for d in data if d[18] == note and d[21] == note_2]))#Ac
    #         print(np.std([d[12] for d in data if d[18] == note and d[21] == note_2]))  # Ac
    #         print(np.mean([d[20] for d in data if d[18] == note and d[21] == note_2]))#Ac
    #         print(np.std([d[20] for d in data if d[18] == note and d[21] == note_2]))  # Ac

    data = [dd for dd in data if dd[19]=='strong']

    # influential_factor_column = 19
    # x_label = 'deceleration and acceleration level'
    # x_label_mark = ['mild', 'strong']

    # influential_factor_column = 16
    # x_label = 'cruise pattern'
    # x_label_mark = ['dip', 'cruise']

    # influential_factor_column = 21
    # x_label = 'headway setting'
    # x_label_mark = ['headway - 1', 'headway - 3']

    # influential_factor_column = 0
    # x_label = 'speed level'
    # x_label_mark = ['low', 'middle', 'high']

    # draw_oscillation_statistics(data,6,2,influential_factor_column,x_label,x_label_mark,'Deceleration starting difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,3,influential_factor_column,x_label,x_label_mark,'Deceleration ending difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,4,influential_factor_column,x_label,x_label_mark,'Acceleration starting difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,5,influential_factor_column,x_label,x_label_mark,'Acceleration ending difference (FV-LV) ($s$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,7,influential_factor_column,x_label,x_label_mark,'Minimum speed difference (FV-LV) ($mph$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,8,influential_factor_column,x_label,x_label_mark,'Ending speed difference (FV-LV) ($mph$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,9,influential_factor_column,x_label,x_label_mark,'Acceleration rate difference (FV-LV) ($m$/$s^2$)',stick_plot=True)
    # draw_oscillation_statistics(data,6,10,influential_factor_column,x_label,x_label_mark,'Deceleration rate difference (FV-LV) ($m$/$s^2$)',stick_plot=True)

    # draw_oscillation_statistics_2(data,6,2,0,16,'Speed Level','Deceleration starting difference (s)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,3,0,16,'Speed Level','Deceleration ending difference (s)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,4,0,16,'Speed Level','Acceleration starting difference (s)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,5,0,16,'Speed Level','Acceleration ending difference (s)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,7,0,16,'Speed Level','Minimum speed difference (mph)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,8,0,16,'Speed Level','Ending speed difference (mph)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,9,0,16,'Speed Level','Acceleration rate difference (m per s2)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,10,0,16,'Speed Level','Deceleration rate difference (m per s2)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,14,0,16,'Speed Level','FV acceleration rate (m per s2)',stick_plot=True)
    # draw_oscillation_statistics_2(data,6,15,0,16,'Speed Level','FV deceleration rate (m per s2)',stick_plot=True)

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

    draw_oscillation_statistics_2_and_more(data,[2,3,4,5],['Deceleration\nstart','Deceleration\nend','Acceleration\nstart','Acceleration\nend'],
                                           influential_factor_column,x_label_mark,x_label,'Delay ($s$)',
                                           stick_plot=True)


    # draw_oscillation_statistics_multiple_val(data, 6, [2,4,5], influential_factor_column, x_label,x_label_mark,
    #                             'Delay ($s$)', y_limit = [0, 15])
    draw_oscillation_statistics_multiple_val(data, 6, [12,14],  influential_factor_column, x_label,x_label_mark,
                                'Acceleration rate ($m$/$s^2$)', y_limit = [0, 2], stick_plot=True)
    draw_oscillation_statistics_multiple_val(data, 6, [13,15],  influential_factor_column, x_label,x_label_mark,
                                'Deceleration rate ($m$/$s^2$)', y_limit = [0, 1.5] ,stick_plot=True)

if __name__ == '__main__':
    oscillation_analyses()
