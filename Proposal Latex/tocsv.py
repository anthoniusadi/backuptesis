import os
import csv
my_data=[]
my_label=[]
path='/home/m448690/ASL4/augmented_data'    
def direktori(pth,dirname):
    for i in os.listdir(path):
        new_path=path+i
        for img  in os.walk(new_path):
            for n in img[2]:
                data=new_path+"/"+n
                label = i
                my_data.append(data)
                my_label.append(label)
        with open (dirname+'.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['data', 'label'])
            for index_data in range(len(my_data)):
                print(my_data[index_data])
                print(*my_label[index_data])
                writer.writerow([my_data[index_data] ,  my_label[index_data]])
direktori(path,"train_augmented")
