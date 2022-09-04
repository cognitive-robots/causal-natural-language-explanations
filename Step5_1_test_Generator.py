#!/usr/bin/env python
# |**********************************************************************;
# Project           : Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import  argparse
import  sys
import  os
import  numpy as np
import  h5py
import csv
import  tensorflow        as      tf
from    collections       import namedtuple
from    src.utils         import  *
from    src.preprocessor_Gen  import  *
from    src.config_VA        import  *
from    src.LSTM_Gen_slim_v2_pos_penalties    	      import  * #TODO Change whether _pos or _w_penalties or pos_penalties or not
from    sys               import platform
from    tqdm              import tqdm
import dataloader_Gen as dataloader
import sys
import json
import pickle
from    src.utils_nlp   import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('--getscore',      type=bool,  default=False, help='get performance scores')
  #parser.add_argument('--showvideo',     type=bool,  default=False, help='show video')
  parser.add_argument('--useCPU',        type=bool,  default=False, help='without GPU processing')
  parser.add_argument('--validation',    type=bool,  default=False, help='use validation set')
  parser.add_argument('--gpu_fraction',  type=float, default=0.7,   help='GPU usage limit')
  parser.add_argument('--extractText',   type=bool,  default=True,  help='extract attention maps')
  args = parser.parse_args(args)
  print_captions = True
  np.set_printoptions(threshold=99999)

  if platform == 'linux':
    timestamp = "20220513-212213"
    args.model = "./model/LSTM_Gen/" + timestamp + "/model.ckpt"
    args.savepath = "./result/LSTM_Gen/"
    config.timelen = 3500 + 3
    timelen = 3500
    config.batch_size_gen = 1
    savepath_dict = "cap"
  else:
    raise NotImplementedError

  if args.getscore:    check_and_make_folder(args.savepath)
  if args.extractText: check_and_make_folder(config.h5path + "extracted_text/")

  # Create VA model
  Gen_model = LSTM_Gen(alpha_c=config.alpha_c, dropout=False)
  logits_softmax = Gen_model.inference()


  if args.useCPU: # Use CPU only
    tfconfig = tf.ConfigProto( device_count={'GPU':0}, intra_op_parallelism_threads=1)
    sess = tf.Session(config=tfconfig)
  else: # Use GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


  # Load the pretrained model
  saver = tf.train.Saver()
  if args.model is not None:
    saver.restore(sess, args.model)
    print("\rLoaded the pretrained model: {}".format(args.model))

  with open(os.path.join(config.h5path + '{}/{}/idx_to_word.pkl'.format(savepath_dict, 'train')), 'rb') as f:
    idx_to_word = pickle.load(f)

  with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(savepath_dict, 'train')), 'rb') as f:
    word_to_idx = pickle.load(f)

  data_val = dataloader.data_iterator(test=True, model="VA")  #validation=True

  seq_ls, cap_gt_ls, cap_log_ls, accel_gt_ls, accel_ls, c_gt_ls, c_ls = [],[],[],[],[],[],[]
  refs_desc, hypo_desc, refs_just, hypo_just = [], [], [], [] #refs are GT, hypo are logits

  for bv in range(1818): #Size of test set #TODO Change according to dataset.
    alps, context, pred_course, pred_accel, caption, feat, masks, _, _, timestamp, seq_id_b, course_b, accel_b, imgs, _ = next(data_val)

    caption_onehot = (np.arange(config.dict_size) == caption[..., None]).astype(int)
    caption_onehot = np.squeeze(caption_onehot)

    feed_dict = {Gen_model.context: context,
                 Gen_model.pred_acc: pred_accel,
                 Gen_model.pred_course: pred_course,
                 Gen_model.features: feat
                 }

    cap_logits = sess.run([logits_softmax], feed_dict)

    if print_captions == True:
      cap_text = convert_cap_vec_to_text(cap_logits[0][0], idx_to_word)
      print("\n")
      print(cap_text)
      cap_text_gt = convert_cap_vec_to_text(caption_onehot, idx_to_word)
      print(cap_text_gt)

    # Get indices
    start_desc, end_desc, start_just, end_just = get_rel_indices(cap_logits[0][0], word_to_idx)

    # Convert idx to words
    if start_desc != -1: # check if sentence has a separator
      cap_text_desc = convert_cap_vec_to_text(cap_logits[0][0][start_desc:end_desc+1], idx_to_word)
      print(cap_text_desc)
      cap_text_just = convert_cap_vec_to_text(cap_logits[0][0][start_just:end_just+1], idx_to_word)
      print(cap_text_just)

      # Save sentences in lists
      hypo_desc.append(str(cap_text_desc))
      hypo_just.append(str(cap_text_just))

      # Get indices GT
      start_desc_gt, end_desc_gt, start_just_gt, end_just_gt = get_rel_indices(caption_onehot, word_to_idx)
      cap_text_desc_gt = convert_cap_vec_to_text(caption_onehot[start_desc_gt:end_desc_gt + 1], idx_to_word)
      cap_text_just_gt = convert_cap_vec_to_text(caption_onehot[start_just_gt:end_just_gt + 1], idx_to_word)
      print(cap_text_desc_gt)
      print(cap_text_just_gt)

      refs_desc.append(str(cap_text_desc_gt))
      refs_just.append(str(cap_text_just_gt))
    else: # if no separator generated, still append something to align with generated images. Should decrease score.
      hypo_desc.append("NULL")
      hypo_just.append("NULL")
      refs_desc.append("NOT-VALID")
      refs_just.append("NOT-VALID")

    # Save Imgs with Attn
    img_att_np = visualize_attnmap_2(alps[0, 9, :], imgs[0, 9, :, :, :]) #0, 9 #TODO Change for BDDx or SAX
    img_att = Image.fromarray(img_att_np)
    set_n = "test/"
    img_att.save("/root/Workspace/explainable-deep-driving-master/data/processed_full/img_w_attn/" + str(set_n) + str(bv) + ".png") #TODO change SAX/ full...

    seq_ls.append(str(np.squeeze(seq_id_b)))
    cap_gt_ls.append(str(np.squeeze(caption_onehot)))
    cap_log_ls.append(str(np.squeeze(cap_logits)))
    accel_gt_ls.append(str(np.squeeze(accel_b)))
    accel_ls.append(str(np.squeeze(pred_accel)))
    c_gt_ls.append(str(np.squeeze(course_b)))
    c_ls.append(str(np.squeeze(pred_course)))


  if args.extractText:
    with open(config.h5path + "extracted_text/" + "output.csv", 'w', newline='') as out:
      csv.writer(out, delimiter=' ').writerows(zip(seq_ls, cap_gt_ls, cap_log_ls, accel_gt_ls, accel_ls, c_gt_ls, c_ls))

  # Save Lists
  refs_just = [refs_just]
  refs_desc = [refs_desc]
  with open(config.h5path + "extracted_text/" + 'refs_just.pkl', 'wb') as f:
    pickle.dump(refs_just, f)
  with open(config.h5path + "extracted_text/" + 'refs_desc.pkl', 'wb') as f:
    pickle.dump(refs_desc, f)
  with open(config.h5path + "extracted_text/" + 'hypo_desc.pkl', 'wb') as f:
    pickle.dump(hypo_desc, f)
  with open(config.h5path + "extracted_text/" + 'hypo_just.pkl', 'wb') as f:
    pickle.dump(hypo_just, f)

  # Total Result
  print(bcolors.HIGHL + 'Done' + bcolors.ENDC)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
