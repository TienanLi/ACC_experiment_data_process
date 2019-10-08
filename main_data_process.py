import civic
from side_functions import read_data_from_csv

def main():
    #run 1: 1 set - data in still
    #run 2: 3 set - data in highway
    #run 3: 1 set - example data download from cabana
    #run 4: 1 set - Toyota data
    run=3
    set=1
    messeage_dict=read_data_from_csv('\\data\\output%s_%s.csv' % (run, set))

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
