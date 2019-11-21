from side_functions import read_data_from_csv,get_ID_loc_and_model,read_two_vehicle_data,analyze_and_draw,analyze_and_draw_2
import pickle
import os

def main():
    if run.isdigit():
        ID=int(run)
    else:
        ID=int(run.split('_')[0])
    messeage_ID_location, model = get_ID_loc_and_model(ID)
    if ID<=8 or ID>=11:
        print('set:', set)
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
        print('set:', set)
        analyze_and_draw(messeage_dict, 'prius', run, set)

        # try:
        #     fo=open(os.path.dirname(__file__) + '\\data\\prius\\two_vehicle\\run_%s' % run,'rb')
        #     messeage_dict_all=pickle.load(fo)
        #     fo.close()
        # except:
        #     messeage_dict_all=read_two_vehicle_data('\\data\\prius\\two_vehicle\\',run,'l','f')
        #     fo=open(os.path.dirname(__file__) + '\\data\\prius\\two_vehicle\\run_%s' % run,'wb')
        #     pickle.dump(messeage_dict_all,fo)
        #     fo.close()
        # analyze_and_draw_2(messeage_dict_all,run,'l','f')
    else:
        messeage_dict = read_data_from_csv('\\data\\prius\\three_vehicle\\output%s_%s.csv' % (run, set),
                                           5)
        print('set:', set)
        analyze_and_draw(messeage_dict, 'prius', run, set)
    #     try:
    #         fo=open(os.path.dirname(__file__) + '\\data\\prius\\three_vehicle\\run_%s' % run,'rb')
    #         messeage_dict_all=pickle.load(fo)
    #         fo.close()
    #     except:
    #         messeage_dict_all=read_two_vehicle_data('\\data\\prius\\three_vehicle\\',run,'m','f')
    #         fo=open(os.path.dirname(__file__) + '\\data\\prius\\three_vehicle\\run_%s' % run,'wb')
    #         pickle.dump(messeage_dict_all,fo)
    #         fo.close()
    #     analyze_and_draw_2(messeage_dict_all,run,'m','f')

if __name__ == '__main__':
    global run
    run='13'
    global set
    for set in range(3,4):
        main() #The beginning of the program. It goes to the def main() function,
