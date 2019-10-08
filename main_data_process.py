import civic
import prius
from side_functions import read_data_from_csv

def main():
    #run 1: 1 set - data in still, message ID at 1
    #run 2: 3 set - data in highway, message ID at 1
    #run 3: 1 set - example data download from cabana, message ID at 1
    #run 4: 1 set - Toyota data, message ID at 5
    run=4
    set=1
    messeage_ID_location=1
    model='civic'
    if run==4:
        messeage_ID_location=5
        model='prius'
    messeage_dict=read_data_from_csv('\\data\\%s\\output%s_%s.csv' % (model,run, set),
                                     messeage_ID_location)
    # analyze_civic(messeage_dict)
    analyze_prius(messeage_dict)


def analyze_prius(messeage_dict):
    # STEER_ANGLE_SENSOR = messeage_dict[37]
    # prius.analyze_STEER_ANGLE_SENSOR(STEER_ANGLE_SENSOR)

    SPEED=messeage_dict[180]
    prius.analyze_SPEED(SPEED)

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
    main() #The beginning of the program. It goes to the def main() function,
