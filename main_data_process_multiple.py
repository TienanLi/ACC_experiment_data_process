from side_function_for_multiple import read_two_vehicle_data,analyze_and_draw_2
import pickle
import os

def main():
    if run.isdigit():
        ID=int(run)
    else:
        ID=int(run.split('_')[0])
    if ID==9:
        try:
            fo=open(os.path.dirname(__file__) + '\\data\\prius\\two_vehicle\\run_%s' % run,'rb')
            messeage_dict_all=pickle.load(fo)
            fo.close()
        except:
            messeage_dict_all=read_two_vehicle_data('\\data\\prius\\two_vehicle\\',run,'l','f')
            fo=open(os.path.dirname(__file__) + '\\data\\prius\\two_vehicle\\run_%s' % run,'wb')
            pickle.dump(messeage_dict_all,fo)
            fo.close()
        analyze_and_draw_2(messeage_dict_all,run,'l','f')
    else:
        try:
            fo=open(os.path.dirname(__file__) + '\\data\\prius\\three_vehicle\\run_%s' % run,'rb')
            messeage_dict_all=pickle.load(fo)
            fo.close()
        except:
            messeage_dict_all=read_two_vehicle_data('\\data\\prius\\three_vehicle\\',run,'m','f')
            fo=open(os.path.dirname(__file__) + '\\data\\prius\\three_vehicle\\run_%s' % run,'wb')
            pickle.dump(messeage_dict_all,fo)
            fo.close()
        analyze_and_draw_2(messeage_dict_all,run,'m','f')

if __name__ == '__main__':
    global run
    for run in [str(i) for i in range(9,11)]:
        main() #The beginning of the program. It goes to the def main() function,
