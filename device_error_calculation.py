from GPS_data_functions import read_data_from_summary_csv
import matplotlib.pyplot as plt
import numpy as np

def main(folder_name):
    traj_dict = read_data_from_summary_csv(folder_name, 3)
    speedDiff = [abs(traj_dict[0][1][i] - traj_dict[0][2][i]) for i in range(len(traj_dict[0][1])) if traj_dict[0][1][i] > 20 and abs(traj_dict[0][1][i] - traj_dict[0][2][i]) < 1 and abs(traj_dict[0][4][i] - traj_dict[0][5][i]) < 5]
    print('speed:', np.percentile(speedDiff, 50))
    locDiff = [abs(traj_dict[0][4][i] - traj_dict[0][5][i]) for i in range(len(traj_dict[0][1])) if traj_dict[0][1][i] > 20 and abs(traj_dict[0][1][i] - traj_dict[0][2][i]) < 1 and abs(traj_dict[0][4][i] - traj_dict[0][5][i]) < 5]
    print('location:',np.percentile(locDiff, 50))
    # plt.hist(speedDiff)
    # plt.show()

if __name__=='__main__':
    main('/platooned_data/04-25-2020/')

