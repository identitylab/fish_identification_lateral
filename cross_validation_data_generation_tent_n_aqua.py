from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import glob
import os
import random
import numpy as np
from shutil import copy,rmtree,copytree,copy2
import math
import sys
import argparse
from glob import glob

#parser = argparse.ArgumentParser(description='Split the data and generate the train and test set')

#parser.add_argument('byIDorByImages', help='split the data by ID, or by images, if by images, all IDs are used in training set, default is by ID, 1 means split by ID, 0 or otherwise means by images', type=int)
#parser.add_argument('train_weight', help='the ratio of images taken out to the trining set, default is 50%', type=float)

directory_str_tent = 'data/SESSION_TENT_NEW/SESSION1_LT'
directory_str_aqua = 'data/SESSION_AQUARIUM/SESSION1_LT'
ST_DIR_TENT = 'data/SESSION_TENT_NEW/SESSION1_ST/'
ST_DIR_AQUA = 'data/SESSION_AQUARIUM/SESSION1_ST/'
#train_dir = 'tmp_tent/train/'


def addPrefix(path,prefix):
    for root, subdirs, files in os.walk(path):
        for name in files:
            curr_fld = os.path.basename(root)
            oldname = os.path.join(path, curr_fld, name)
            splt_name = name.split('.')
            myname = '_'.join([prefix, splt_name[0][-1], splt_name[0], curr_fld + '.' + splt_name[1]])
            newname = os.path.join(path, curr_fld, myname)
            os.rename(oldname, newname)
#args = parser.parse_args()
#byIDorByImages = args.byIDorByImages
#train_weight = args.train_weight
#print(byIDorByImages)
def generateDataset(byIDorByImages=True,train_weight=0.5,train_dir_tent='tmp_tent/train/',test_dir_tent='tmp_tent/test/',includeST=True, includeTentnAquaBoth=False):
    test_dir_tent = 'tmp_tent/test/'
    train_dir_aqua = 'tmp_aqua/train/'
    test_dir_aqua = 'tmp_aqua/test/'

    # remove any file exist
    if os.path.exists(train_dir_tent):
        rmtree(train_dir_tent)
        # rmtree(train_dir_aqua)
        rmtree(test_dir_tent)
        # rmtree(test_dir_aqua)

    # check_folder(train_dir)
    check_folder(test_dir_tent)
    check_folder(test_dir_aqua)

    # first copy ST to train
    if includeST:
        copytree(ST_DIR_TENT, train_dir_tent)
        pre = "tent_st"
        addPrefix(train_dir_tent, pre)
        copytree(ST_DIR_AQUA, train_dir_aqua)
        pre = "aqua_st"
        addPrefix(train_dir_aqua, pre)

    if includeTentnAquaBoth:
        copytree(ST_DIR_AQUA, train_dir_tent)
        pre = "aqua"
        addPrefix(train_dir_tent, pre)

    SPLIT_WEIGHTS_INTRA_ID = (
        train_weight,1-train_weight, 0.0)  # train cv val vs test for each identity, 50% are taken as train and 50% as test
    SPLIT_WEIGHTS_INTER_ID = (
        train_weight, 1 - train_weight, 0.0)  # train cv val vs test for each identity, 50% are taken as train and 50% as test

    if byIDorByImages==1:
        train_ID_list, test_ID_list = splitID(directory_str_tent,SPLIT_WEIGHTS_INTRA_ID)
        print(train_ID_list)
        print("above IDs are used in training, remianing will be used for testing, in dir tmp_tent/test/*")

        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION1_LT', train_ID_list,train_dir_tent,test_dir_tent,pre='sess1')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION2', train_ID_list,train_dir_tent,test_dir_tent,pre='sess2')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION3', train_ID_list,train_dir_tent,test_dir_tent,pre='sess3')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION4', train_ID_list,train_dir_tent,test_dir_tent,pre='sess4')


        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION1_LT', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess1')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION2', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess2')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION3', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess3')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION4', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess4')

    else:
        print('All IDs are used in training')




def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def generateDatasetBySplitingIdentity(directory_str,train_list,train_dir,test_dir,pre='sess1'):
    g = os.walk(directory_str)
    for path, dir_list, file_list in g:
        for id_dir in dir_list:
            if train_list.__contains__(id_dir): # in train set, maybe the filename will be same, hence we need to rename each session
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    #print(images)
                    dst_dir = os.path.join(train_dir, id_dir)
                    check_folder(dst_dir)
                    head, tail = os.path.split(images)
                    finalpath=os.path.join(dst_dir, pre+tail)
                    copy(images, finalpath)

            else:
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    dst_dir = os.path.join(test_dir,str.split(directory_str,'/')[-1], id_dir)
                    check_folder(dst_dir)
                    copy(images, dst_dir)


def splitID(directory_str,SPLIT_WEIGHTS_INTRA_ID):
    dir_list = [o for o in os.listdir(directory_str) if os.path.isdir(os.path.join(directory_str, o))]
    ids=len(dir_list)
    train_num = math.floor(ids * SPLIT_WEIGHTS_INTRA_ID[0])
    train_ID_list = random.sample(dir_list, k=train_num)
    test_ID_list=[]
    for anyid in dir_list:
        if not train_ID_list.__contains__(anyid):
            test_ID_list.append(anyid)
    return  train_ID_list,test_ID_list
