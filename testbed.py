from MPE_core import mpe
import csv
import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt

paths = {
    'xml': '../../datasets/INBreast/XML/',
    'dcm': '../../datasets/INBreast/DICOM/',
    'root': '../../datasets/INBreast/',
    'csv': '../../datasets/INBreast/INbreast_compact.csv',
    'dtst': '../../datasets/dataset/single_layer/'
}

args = {
    'p_size': 1024,
    'p_step': 50,
    'ppi': 40,
    'h_ppi': 40 
}

# Build_dcm_dict:
# ---------------
# Given the descriptive csv file of the original
# INBreast dataset, this function generates a dictionary
# with the needed information for each image
# 
# -> path_to_csv: path to the csv file
# <- Dictionary containing the corresponding info

def build_dcm_dict(path_to_csv):
  dcm_list = []

  with open(path_to_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    dict_key   = 0

    for row in csv_reader:
      if line_count == 0:
        line_count += 1
        continue
      obj = {"filename": row[5],
              "acr": row[6],
              "birads": row[7],
              "mass": row[8],
              "calc": row[9]}
      dcm_list.append(obj)
      line_count += 1
      dict_key += 1
    print(f'Processed {line_count} lines.')

  return dcm_list

l = build_dcm_dict(paths['csv'])


non_h = []
for i in range(len(l)):
  if l[i]['mass'] == 'X':
    non_h.append(l[i]['filename'])

print(len(non_h))




import cv2
from tqdm import tqdm

idx = 0
tidx = 0
l = len(non_h)*80
p_dataset = np.zeros((l, 256, 256, 5))

for i in tqdm(range(len(non_h))):
  # try: 
  i_path = non_h[i]
  m_path = non_h[i]
  print(i_path)
  ipe = mpe.ImgPatchExtractor(dn="INBreast", p_size=args["p_size"], hppi=args["h_ppi"], nhppi=args["ppi"], mar=0.01, bar=0.8, info="random/path", img_path=i_path, mask_path=m_path, root_dir=paths['root'])
  
  for w in range(0, ipe.p_idx): #len(ipe.patches)):
    temp = cv2.resize(ipe.patches[w, :, :, 0], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    p_dataset[tidx, :, :, 0] = temp
    temp = cv2.resize(ipe.patches[w, :, :, 1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    p_dataset[tidx, :, :, 1] = temp
    temp = cv2.resize(ipe.patches[w, :, :, 2], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    p_dataset[tidx, :, :, 2] = temp
    temp = cv2.resize(ipe.patches[w, :, :, 3], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    p_dataset[tidx, :, :, 3] = temp
    temp = cv2.resize(ipe.patches[w, :, :, 4], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    p_dataset[tidx, :, :, 4] = temp
    tidx += 1

  idx += 1
  print("Patches extracted: ", tidx)
  # except Exception as error:
  #   exc_type, exc_obj, exc_tb = sys.exc_info()
  #   fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
  #   print(exc_type, fname, exc_tb.tb_lineno)
  #   continue

print(tidx, len(p_dataset))

np.save("../../datasets/proper_datasets/inbreast_1024_256_4040_filt", p_dataset[:tidx, :, :, :])

dataset = np.load("../../datasets/proper_datasets/inbreast_1024_256_4040_filt.npy")



# train = dataset[:4500, :, :, :]
# perm = np.random.permutation(len(train))
# train = train[perm]
# train_fold_1 = train[:1125, :, :, :]
# train_fold_2 = train[1125:2250, :, :, :]
# train_fold_3 = train[2250:3375, :, :, :]
# train_fold_4 = train[3375:, :, :, :]
# np.save("../../datasets/proper_datasets/inbreast_tfold_1", train_fold_1)
# np.save("../../datasets/proper_datasets/inbreast_tfold_2", train_fold_2)
# np.save("../../datasets/proper_datasets/inbreast_tfold_3", train_fold_3)
# np.save("../../datasets/proper_datasets/inbreast_tfold_4", train_fold_4)
# np.save("../../datasets/proper_datasets/inbreast_test", dataset[4500:, :, :, :])
# print(dataset.shape)
# # plt.figure()
# # plt.imshow(dataset[5110, :, :, 2], cmap='gray')
# # plt.savefig("test.png")