import matplotlib.pyplot as plt
import os


def draw_fig(x,x_label,y,y_label):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    plt.plot(x, y)
    plt.xlabel(x_label,fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.savefig(y_label+'.png')
    plt.close()

def hex_to_byte(hex_str, length):
    scale = 16  ## equals to hexadecimal
    num_of_bits = 8 * length
    return bin(int(hex_str, scale))[2:].zfill(num_of_bits)

def read_data_from_csv(file_name):

    information={}
    fo = open(os.path.dirname(__file__)+file_name, 'r')
    fo.readline()
    line_num=0
    while True:
        line_num+=1
        #for each line
        line = fo.readline()
        if not line:
            break
        #split the whole line by comma
        tmp = line.split(',')
        if len(tmp)<4:
            break

        time=line_num
        BUS=tmp[0]
        message_ID=tmp[1]
        message=tmp[2]
        try:
            message_length=int(tmp[3].replace("\n", ""))
        except:
            message_length=0

        if message_ID in information.keys():
            information[message_ID].append((time,message,message_length,BUS))
        else:
            information[message_ID]=[]
            information[message_ID].append((time,message,message_length,BUS))
    fo.close()
    return information