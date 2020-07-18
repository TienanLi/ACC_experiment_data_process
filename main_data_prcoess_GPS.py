import os
from GPS_data_functions import read_data_from_seperated_csv, traj_process, \
    read_data_from_summary_csv, save_info, speed_visulization_2_subplot

def main(folder_name,platoon_number):
    if 'summary_output_1_%s.csv'%platoon_number in os.listdir(os.path.dirname(__file__) + folder_name):
        traj_dict = read_data_from_summary_csv(folder_name, platoon_number)
    else:
        location, start_end_time = read_data_from_seperated_csv(folder_name,2,3,5,1,4)
        traj_dict = traj_process(location, start_end_time, folder_name, platoon_number)
    save_info(traj_dict, folder_name, MA_window=1, extended_period = 65, overall=False)
    # speed_visulization_2_subplot(traj_dict, folder_name, MA_window=1, extended_period = 65, draw_veh=2)

if __name__=='__main__':
    experiment_date = [7,8,12,14,15]
    for date in experiment_date:
        print(date)
        main('/platooned_data/03-%s-2020/'%(str(date).zfill(2)),platoon_number=3)


