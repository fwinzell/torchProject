import os
import pandas as pd

base_dir = "/home/fi5666wi/Documents/Prostate_images/Patches_149"
data_dict = {'Dataset': [],
             'WSIs': [],
             'Patches': []}

wsi_count = 0
pat_count = 0
for dataset in os.listdir(base_dir):
    data_dict['Dataset'].append(dataset)
    data_path = os.path.join(base_dir, dataset, 'patches')
    for wsi in os.listdir(data_path):
        data_dict['WSIs'].append(wsi)
        wsi_count += 1
        npatches = len([img for img in os.listdir(os.path.join(data_path, wsi))])
        data_dict['Patches'].append(npatches)
        pat_count += npatches

print(data_dict)
print('Total wsis: {}'.format(wsi_count))
print('Total patches: {}'.format(pat_count))

