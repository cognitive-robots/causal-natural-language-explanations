#!/usr/bin/env python
# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import  os
import  h5py
from    src.config_VA      import  *
from    src.utils_nlp   import *
from    src.utils       import *
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #-----------------------
    # Parameters
    #-----------------------
    if platform == 'linux':
        param = dict2(**{
            "max_length":       20,    # the maximum length of sentences
            "vid_max_length":   10,    # the maximum length of video sequences
            "size_of_dict":     config.dict_size, # the size of dictionary
            "chunksize":        4,    # for h5 format file writing, 10
            "savepath":         'cap',
            "FINETUNING":       False,
            "SAVEPICKLE":       True })
    else:
        raise NotImplementedError
    
    check_and_make_folder(config.h5path+"cap/log/")
    check_and_make_folder(config.h5path+"cap/feat/")

    #-----------------------
    # For each split, collect feats/logs
    #-----------------------
    #for split in ['train', 'test', 'val']:
    context_shapes = []
    cap_shapes = []
    data_samples = []
    for split in ['train', 'val', 'test']:
        check_and_make_folder(config.h5path+param.savepath+'/'+split)

        # Step1: Preprocess caption data + refine captions
        caption_file = config.h5path + 'captions_BDDX_' + split + '.json'
        #TODO change process_caption_data to process_caption_data_w_pos for PoS prediction
        annotations, pos_list  = process_caption_data_w_pos(caption_file=caption_file, image_dir=config.h5path+'feat/', max_length=param.max_length)
        if param.SAVEPICKLE: save_pickle(annotations, config.h5path + '{}/{}/{}.annotations.pkl'.format(param.savepath, split, split))
        print(bcolors.BLUE   + '[main] Length of {} : {}'.format(split, len(annotations)) + bcolors.ENDC)

        # Step2: Build dictionary
        if param.FINETUNING:
            with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                word_to_idx = pickle.load(f)
        else:
            if split == 'train':
                word_to_idx, idx_to_word = build_vocab(annotations=annotations, size_of_dict=param.size_of_dict)
                if param.SAVEPICKLE: save_pickle(word_to_idx, config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, split))
                if param.SAVEPICKLE: save_pickle(idx_to_word, config.h5path + '{}/{}/idx_to_word.pkl'.format(param.savepath, split))
            else:
                with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                    word_to_idx = pickle.load(f)

        # Step3: Clustering
        if split == 'train': clusters, ind_cluster = cluster_annotations(annotations=annotations, k=20)

        # Step4: word to index
        #  #TODO Diff Fct with POS or without
        captions, pos = build_caption_vector_w_pos(annotations=annotations, pos_ls=pos_list, word_to_idx=word_to_idx, max_length=param.max_length)
        if param.SAVEPICKLE: save_pickle(captions, config.h5path + '{}/{}/{}.captions.pkl'.format(param.savepath, split, split))

        # Step5: feat & masks #TODO change build_feat_matrix (BDDX) or build_feat_matrix_SAX (SAX)
        all_feats4Cap, all_masks4Cap, all_logs, all_attns4Cap, all_contexts4cap, all_imgs4Cap, nr_samples_with_data = build_feat_matrix_SAX(
                                                           annotations=annotations, 
                                                           max_length=param.vid_max_length, 
                                                           fpath=config.h5path, 
                                                           FINETUNING=param.FINETUNING)

        # Step6: Saving these data into hdf5 format
        feat = h5py.File(config.h5path + "cap/feat/" + split + ".h5", "w")
        logs = h5py.File(config.h5path + "cap/log/"  + split + ".h5", "w")

        dset = feat.create_dataset("/X",     data=all_feats4Cap, chunks=(param.chunksize, param.vid_max_length, 64, 12, 20) ) #fc8
        dset = feat.create_dataset("/mask",  data=all_masks4Cap)
        dset = feat.create_dataset("/img",   data=all_imgs4Cap, chunks=(param.chunksize, param.vid_max_length, 90, 160, 3))
        
        dset = logs.create_dataset("/attn",  data=all_attns4Cap, chunks=(param.chunksize, param.vid_max_length, 240))
        dset = logs.create_dataset("/context", data=all_contexts4cap, chunks=(param.chunksize, param.vid_max_length, 64))
        dset = logs.create_dataset("/Caption",      data=captions)
        dset = logs.create_dataset("/POS", data=pos)
        dset = logs.create_dataset("/timestamp",    data=all_logs['timestamp'])
        dset = logs.create_dataset("/curvature",    data=all_logs['curvature'])
        dset = logs.create_dataset("/accelerator",  data=all_logs['accelerator'])
        dset = logs.create_dataset("/speed",        data=all_logs['speed'])
        dset = logs.create_dataset("/course",       data=all_logs['course'])
        dset = logs.create_dataset("/goaldir",      data=all_logs['goaldir'])
        dset = logs.create_dataset("/pred_accel",   data=all_logs['pred_accel'])
        dset = logs.create_dataset("/pred_courses" ,data=all_logs['pred_courses'])

        if split == 'train': dset = logs.create_dataset("/cluster",      data=ind_cluster)
        print(all_contexts4cap.shape)
        print(captions.shape)
        print(all_logs['pred_accel'].shape)
        context_shapes.append(all_contexts4cap.shape[0])
        cap_shapes.append(captions.shape[0])
        data_samples.append(nr_samples_with_data)
        print(bcolors.GREEN + '[main] Finish writing into hdf5 format: {}'.format(split) + bcolors.ENDC)

    print(idx_to_word)
    print(context_shapes)
    print(cap_shapes)
    print(data_samples)

if __name__ == "__main__":
    main()
