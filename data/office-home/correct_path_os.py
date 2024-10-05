import re
for name in ['Real_World','Product','Clipart','Art']:

    with open(name+'.txt','r') as f_in, open(name+'_t_ordered.txt','a') as f_out:
        old=0
        for line in f_in:
           
            m=re.search('\s\d+', line)
            class_start_char=line[line.index(name)+len(name)+1]
            
            if class_start_char in ['A','B','C','D','E','F']:# shared classes
                if old==0:
                    label=0
                elif line[line.index(name)+len(name)+1:line.index(name)+len(name)+1+4] !=old and old!=0:
                    label=label+1
                print(line[line.index(name)+len(name)+1:line.index(name)+len(name)+1+4],old,label)
                f_out.write('data/office-home/OfficeHome_os'+line[line.index('/images'):line.index('.jpg')+4 ]+' '+str(label)+'\n')

                old=line[line.index(name)+len(name)+1:line.index(name)+len(name)+1+4]
            else: # only target classes
                f_out.write('data/office-home/OfficeHome_os'+line[ line.index('/images'):line.index('.jpg')+4 ]+' 25\n')






    # with open('old.txt') as f_in, open("new.txt", "a") as f_out:
    #     for line in f_in:
    #         a, b = line.split()
    #         f_out.write("('{}', '{}')\n".format(a, b))

