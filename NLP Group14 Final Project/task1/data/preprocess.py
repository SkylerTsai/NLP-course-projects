import csv
import os

for split in ['train', 'dev', 'test']:
    inputs = []
    path = split + '.csv'
    with open(path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i != 0:
                lst =[]
                lst.append(row[0])
                lst.append(row[1])
                if split != 'test':
                    for j in range(2, 20):
                        if row[j] == '-2':
                            lst.append(0)
                        else:
                            lst.append(1)
                else:
                    for j in range(2, 20):
                        lst.append(0)
                inputs.append(lst)

    path = '_' + split + '.csv'
    with open(path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id',
            'review',
            'Location#Transportation',
            'Location#Downtown',
            'Location#Easy_to_find',
            'Service#Queue',
            'Service#Hospitality',
            'Service#Parking',
            'Service#Timely',
            'Price#Level',
            'Price#Cost_effective',
            'Price#Discount',
            'Ambience#Decoration',
            'Ambience#Noise',
            'Ambience#Space',
            'Ambience#Sanitary',
            'Food#Portion',
            'Food#Taste',
            'Food#Appearance',
            'Food#Recommend',
        ])
        for row in inputs:
            writer.writerow(row)
                
