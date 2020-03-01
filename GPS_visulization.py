import os
from GPS_data_functions import read_data_from_seperated_csv, traj_process, \
    read_data_from_summary_csv, speed_visulization

def main(folder_name):
    if 'summary_output_1.csv' in os.listdir(os.path.dirname(__file__) + folder_name):
        traj_dict = read_data_from_summary_csv(folder_name)
    else:
        location, start_end_time = read_data_from_seperated_csv(folder_name,2,3,5,1)
        traj_dict = traj_process(location, start_end_time, folder_name)
    speed_visulization(traj_dict)


if __name__=='__main__':
    main('/platooned_data/02-25-2020/')

