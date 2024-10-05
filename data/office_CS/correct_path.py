import re
for name in ['amazon_list','dslr_list','webcam_list']:

    with open(name+'.txt','r') as f_in, open(name+'_t.txt','a') as f_out:
        for line in f_in:
            m=re.search('\s\d+', line)
            class_num=int(line[m.start():m.end()])

            if class_num<10:# shared classes
                f_out.write(line [0:11]+'_OS'+line[11:])

            elif class_num>19: # only target classes
                print(line[11:m.start()+1])
                f_out.write(line[0:11]+'_OS'+line[11:m.start()+1]+'10\n')






    # with open('old.txt') as f_in, open("new.txt", "a") as f_out:
    #     for line in f_in:
    #         a, b = line.split()
    #         f_out.write("('{}', '{}')\n".format(a, b))

