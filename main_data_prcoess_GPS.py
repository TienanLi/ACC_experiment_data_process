import os
from GPS_data_functions import read_data_from_seperated_csv, traj_process, \
    read_data_from_summary_csv, speed_visulization

def main(folder_name,platoon_number):
    if 'summary_output_1_%s.csv'%platoon_number in os.listdir(os.path.dirname(__file__) + folder_name):
        traj_dict = read_data_from_summary_csv(folder_name, platoon_number)
    else:
        location, start_end_time = read_data_from_seperated_csv(folder_name,2,3,5,1)
        traj_dict = traj_process(location, start_end_time, folder_name, platoon_number)
    speed_visulization(traj_dict, folder_name, overall=True)


if __name__=='__main__':
    main('/platooned_data/03-15-2020/',platoon_number=3)

