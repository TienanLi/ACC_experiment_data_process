from Analyze_functions import read_data_from_csv,analyze_and_draw
from base_functions import get_ID_loc_and_model
import pickle
import os

def main():
    if run.isdigit():
        ID=int(run)
    else:
        ID=int(run.split('_')[0])
    messeage_ID_location, model = get_ID_loc_and_model(ID)
    if ID<=8 or ID>=11:
        try:
            fo=open(os.path.dirname(__file__) + '\\data\\%s\\run_%s_set_%s'  % (model, run, set),'rb')
            messeage_dict=pickle.load(fo)
            fo.close()
        except:
            messeage_dict = read_data_from_csv('\\data\\%s\\output%s_%s.csv' % (model, run, set),
                                               messeage_ID_location)
            fo=open(os.path.dirname(__file__) + '\\data\\%s\\run_%s_set_%s'  % (model, run, set),'wb')
            pickle.dump(messeage_dict,fo)
        analyze_and_draw(messeage_dict, model, run, set)

    elif ID==9:
        messeage_dict = read_data_from_csv('\\data\\prius\\two_vehicle\\output%s_%s.csv' % (run, set),
                                           5)
        analyze_and_draw(messeage_dict, 'prius', run, set)
    else:
        messeage_dict = read_data_from_csv('\\data\\prius\\three_vehicle\\output%s_%s.csv' % (run, set),
                                           5)
        analyze_and_draw(messeage_dict, 'prius', run, set)

if __name__ == '__main__':
    global run
    global set
    for run in ['9_f','10_m']:
        for set in range(1,20):
            try:
                main() #The beginning of the program. It goes to the def main() function,
            except:
                print('this set not exists')
            # main()
