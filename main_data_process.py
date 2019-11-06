import civic
import prius
from side_functions import read_data_from_csv,draw_traj

def main():
    #run 1: 1 set - data in still, message ID at 1
    #run 2: 3 set - data in highway, message ID at 1
    #run 3: 1 set - example data download from cabana, message ID at 1
    #run 4: 1 Toyota data, message ID at 5: set 3,4,5 preliminary; set 6,7,8 large headway; 9 medium headway；10，11，12,13,14 small headway
    messeage_ID_location=5
    model='prius'
    if run==5:
        model='carolla'
    if run<=3:
        model='civic'
        messeage_ID_location=1

    messeage_dict=read_data_from_csv('\\data\\%s\\output%s_%s.csv' % (model,run, set),
                                     messeage_ID_location)
    # analyze_civic(messeage_dict)
    analyze_prius(messeage_dict,model)


def analyze_prius(messeage_dict,model_name):
    # STEER_ANGLE_SENSOR = messeage_dict[37]
    # prius.analyze_STEER_ANGLE_SENSOR(STEER_ANGLE_SENSOR)

    LEAD_INFO=messeage_dict[466]
    ACC_using_ts,ACC_using=prius.analyze_PCM_CRUISE(LEAD_INFO)
    # LEAD_INFO=messeage_dict[467]
    # ACC_ready_ts,ACC_ready=prius.analyze_PCM_CRUISE_2(LEAD_INFO)

    SPEED=messeage_dict[180]
    speed_time_series,speed=prius.analyze_SPEED(SPEED)
    LEAD_INFO=messeage_dict[742]
    LEAD_INFO_time_series,front_space,relative_speed=prius.analyze_LEAD_INFO(LEAD_INFO)

    draw_traj(speed_time_series,speed,LEAD_INFO_time_series,front_space,relative_speed,'figures/'+str(run)+'_'+str(set))


def analyze_civic(messeage_dict):
    ENGINE_DATA=messeage_dict['0x158']
    civic.analyze_ENGINE_DATA(ENGINE_DATA)
    KINEMATICS = messeage_dict['0x94'] # this is to get longi_accel
    civic.analyze_KINEMATICS(KINEMATICS)
    # GAS_PEDAL_2=messeage_dict['0x130']
    # civic.analyze_GAS_PEDAL_2(GAS_PEDAL_2)
    # POWERTRAIN_DATA=messeage_dict['0x17c']
    # civic.analyze_POWERTRAIN_DATA(POWERTRAIN_DATA)
    # STANDSTILL=messeage_dict['0x1b0']
    # civic.analyze_STANDSTILL(STANDSTILL)
    # SEATBELT_STATUS=messeage_dict['0x305']
    # civic.analyze_SEATBELT_STATUS(SEATBELT_STATUS)

if __name__ == '__main__':
    global run
    run=4
    # global set
    # set=2

    for set in range(4,14):
      main() #The beginning of the program. It goes to the def main() function,
