# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import  json
import  pandas      as      pd
from    src.utils   import  *
import  os
import  collections
from    sklearn.cluster import KMeans
from    sklearn.feature_extraction.text import TfidfVectorizer
from    collections     import Counter
import  numpy           as np
import  h5py
#import  cPickle         as     pickle
import pickle
import sys
from nltk import pos_tag, word_tokenize
import tensorflow as tf


def process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    data = []
    for annotation in caption_data['annotations']:
        annotation['caption'] = annotation['action'] + ' <SEP> ' + annotation['justification'] # ADD separator
        data += [annotation]

    caption_data = pd.DataFrame.from_dict(data)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace("'s"," 's").replace("'ve", " 've").replace("n't", " n't").replace("'re", " 're").replace("'d", " 'd").replace("'ll", " 'll")
        caption = caption.replace('.','').replace(',','').replace('"','').replace("'","").replace("`","")
        caption = caption.replace('&','and').replace('(',' ').replace(')',' ').replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print( bcolors.BLUE + "[_process_caption_data] The number of captions before deletion: %d" %len(caption_data) + bcolors.ENDC ) 
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print( bcolors.BLUE + "[_process_caption_data] The number of captions after deletion: %d" %len(caption_data) + bcolors.ENDC ) 

    return caption_data


def process_caption_data_w_pos(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    data = []
    for annotation in caption_data['annotations']:
        annotation['caption'] = annotation['action'] + ' <SEP> ' + annotation['justification']  # ADD separator
        data += [annotation]

    caption_data = pd.DataFrame.from_dict(data)

    del_idx = []
    pos_dict = {'ADJ':1, 'ADP':2, 'ADV': 3, 'CONJ':4, 'DET':5, 'NOUN':6, 'NUM':7, 'PRT':8, 'PRON':9, 'VERB':10,
                '.':11, 'X':12}
    pos_int_ls = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace("'s", " 's").replace("'ve", " 've").replace("n't", " n't").replace("'re",
                                                                                                     " 're").replace(
            "'d", " 'd").replace("'ll", " 'll")
        caption = caption.replace('.', '').replace(',', '').replace('"', '').replace("'", "").replace("`", "")
        caption = caption.replace('&', 'and').replace('(', ' ').replace(')', ' ').replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
        else:
            # Part of Speech Tagging
            #text = word_tokenize("And now for something completely different")
            caption_pos = caption.replace('<SEP>', ';') #such that it's tokenized as only one position and .
            words = caption_pos.split(" ")
            pos = pos_tag(words, tagset='universal')
            #pos = pos_tag(word_tokenize(caption_pos), tagset='universal')
            #print(pos)
            pos_int = []
            pos_int.append(11)
            for i in range(len(pos)):
                #print(pos[i][1])
                pos_int.append(pos_dict[pos[i][1]])
            pos_int.append(11)
            #print(pos_int)
            pos_int_ls.append(pos_int)
            #sys.exit("Print")

    # delete captions if size is larger than max_length
    print(bcolors.BLUE + "[_process_caption_data] The number of captions before deletion: %d" % len(
        caption_data) + bcolors.ENDC)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print(bcolors.BLUE + "[_process_caption_data] The number of captions after deletion: %d" % len(
        caption_data) + bcolors.ENDC)

    return caption_data, pos_int_ls


def build_vocab(annotations, size_of_dict=10000):
    print(bcolors.GREEN + '[_build_vocab] Build a vocabulary' + bcolors.ENDC)

    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    # limit the size of dictionary
    counter = counter.most_common(size_of_dict)

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>':3}
    idx_to_word = {0: u'<NULL>', 1: u'<START>', 2: u'<END>', 3:u'<UNK>'}
    idx = 3+1

    for word in counter:
        word_to_idx[word[0]] = idx
        idx_to_word[idx]     = word[0]
        idx += 1

    print(bcolors.BLUE + '[_build_vocab] Max length of caption: {}'.format(max_len)   + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_vocab] Size of dictionary: {}'.format(size_of_dict) + bcolors.ENDC)

    return word_to_idx, idx_to_word


def build_caption_vector(annotations, word_to_idx, max_length=15):
    print(bcolors.GREEN + '[_build_caption_vector] String caption -> Indexed caption' + bcolors.ENDC)

    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            else:
                cap_vec.append(word_to_idx['<UNK>'])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)

    print(bcolors.BLUE + '[_build_caption_vector] Building caption vectors' + bcolors.ENDC)

    return captions


def build_caption_vector_w_pos(annotations, pos_ls, word_to_idx, max_length=15):
    print(bcolors.GREEN + '[_build_caption_vector] String caption -> Indexed caption' + bcolors.ENDC)

    if len(annotations) != len(pos_ls):
        sys.exit("Unequal number of captions and pos sentences!")
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)
    pos = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")  # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            else:
                cap_vec.append(word_to_idx['<UNK>'])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])
                pos_ls[i].append(11)

        captions[i, :] = np.asarray(cap_vec)

        pos[i, :] = np.asarray(pos_ls[i])



    print(bcolors.BLUE + '[_build_caption_vector] Building caption vectors' + bcolors.ENDC)

    return captions, pos

def convert_cap_vec_to_text(cap_vec, idx_to_word):
    # input: cap_vec of one sequence
    #for i in range(cap_vec.shape[0]): # for each word
    indices = np.argmax(cap_vec, axis=1)
    sentence = []
    for i in range(indices.shape[0]): #for each word
        sentence.append(idx_to_word[indices[i]])
    text = ' '.join(sentence)
    return text

def convert_cap_vec_to_text_tf(cap_vec, idx_to_word):
    # input: cap_vec of one sequence
    #for i in range(cap_vec.shape[0]): # for each word
    indices = tf.math.argmax(cap_vec, axis=1, output_type=tf.dtypes.int32)
    keys = tf.convert_to_tensor(list(idx_to_word.keys()), dtype=tf.int32)
    values = tf.convert_to_tensor(list(idx_to_word.values()))
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "DEFAULT"
    )
    sentence = []
    for i in range(indices.shape[0]): #for each word
        sentence.append(table.lookup(indices[i]))
    text = tf.strings.join(sentence, separator=' ')
    return text

def get_rel_indices(caption_onehot, word_to_idx):
    # Get indices for action description
    indices = np.argmax(caption_onehot, axis=1)

    first_sep = 0
    end_desc = -1

    start_desc = 1
    #########
    for i in range(22):  # max sentence length + start and end
        if indices[i] == 1:
            start_desc = i + 1
        if indices[i] == word_to_idx['<sep>'] and first_sep == 0:
            end_desc = i - 1
            first_sep += 1

    if end_desc == -1:
        for i in range(22):
            if indices[i] == word_to_idx['because'] and first_sep == 0:
                end_desc = i - 1
                first_sep += 1

        if end_desc == -1:
            print("Skip sample because of no separator")
            return -1, -1, -1, -1

    # Get indices for justification
    end_just = -1
    start_just = -1
    first_sep = 0
    for i in reversed(range(22)):
        if indices[i] == 2:
            end_just = i - 1
        if indices[i] == word_to_idx['<sep>'] and first_sep == 0:
            start_just = i + 1
            first_sep += 1
    if end_just == -1:
        for i in reversed(range(22)):
            if indices[i] == 0:
                end_just = i - 1
    if start_just == -1:
        for i in reversed(range(22)):
            if indices[i] == word_to_idx['because']:
                start_just = i

    return start_desc, end_desc, start_just, end_just

def build_file_names(annotations):
    image_file_names    = []
    id_to_idx           = {}
    idx                 = 0
    image_ids           = annotations['video_id']
    file_names          = annotations['vidName']

    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx

def build_image_idxs(annotations, id_to_idx):
    image_idxs  = np.ndarray(len(annotations), dtype=np.int32)
    image_ids   = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


#------------------------------------------------------|| For Video preprocessing
def pad_video(video_feature, dimension):
    '''
    Fill pad to video to have same length.
    Pad in Left.
    video = [pad,..., pad, frm1, frm2, ..., frmN]
    '''
    padded_feature  = np.zeros(dimension)
    max_length      = dimension[0]
    current_length  = video_feature.shape[0]
    num_padding     = max_length - current_length

    if num_padding == 0:
        padded_feature  = video_feature
    elif num_padding < 0:
        steps           = np.linspace(0, current_length, num=max_length, endpoint=False, dtype=np.int32)
        padded_feature  = video_feature[steps]
    else:
        padded_feature[num_padding:] = video_feature

    return padded_feature


def fill_mask(max_length, current_length, zero_location='LEFT'):
    num_padding = max_length - current_length
    if num_padding <= 0:
        mask = np.ones(max_length)
    elif zero_location == 'LEFT':
        mask = np.ones(max_length)
        for i in range(num_padding):
            mask[i] = 0
    elif zero_location == 'RIGHT':
        mask = np.zeros(max_length)
        for i in range(current_length):
            mask[i] = 1

    return mask



def build_feat_matrix(annotations, max_length, fpath, hz=10, sampleInterval=5, FINETUNING=False):
    print(bcolors.GREEN + '[_build_feat_matrix] Collect feats and masks' + bcolors.ENDC)

    n_examples    = len(annotations) #number of sequences
    max_length_vid= max_length*sampleInterval

    all_logs    = {}
    all_logs['speed']       = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['course']      = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['accelerator'] = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['curvature']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['goaldir']     = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['timestamp']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_accel']  = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_courses']= np.ndarray([n_examples, max_length], dtype=np.float32)


    all_masks4Cap = np.ndarray([n_examples, max_length], dtype=np.float32)  
    #all_feats4Cap = np.memmap('/data2/tmp.dat', dtype='float32', mode='w+', shape=(n_examples, max_length, 64, 12, 20))
    all_feats4Cap = np.ndarray([n_examples, max_length, 64, 12, 20], dtype=np.float32)
    all_imgs4Cap = np.ndarray([n_examples, max_length, 90, 160, 3], dtype=np.uint8)
    all_attns4Cap = np.ndarray([n_examples, max_length, 12*20], dtype=np.float32)
    all_contexts4Cap = np.ndarray([n_examples, max_length, 64], dtype=np.float32)

    sTimes       = annotations['sTime']      # staring timestamp
    eTimes       = annotations['eTime']      # ending timestamp
    vidNames     = annotations['vidName']    # video clip name
    video_ids    = annotations['video_id']   # index of video

    idx          = 0
    samples_with_data = 0
    for sTime, eTime, vidName, video_id in zip(sTimes, eTimes, vidNames, video_ids): # For each caption

        print(bcolors.BLUE + '[_build_feat_matrix] vidName: {}'.format(vidName) + bcolors.ENDC)

        # load feats
        #if (os.path.isfile(fpath+"feats_%s/"%(dataset)+'{}_{}'.format(video_id, vidName)+".h5")) == False: continue 
        if (os.path.isfile(fpath+"log/"+'{}_{}'.format(video_id, vidName) +".h5")) == False: continue 

        feats = h5py.File(fpath+"feat/"+'{}_{}'.format(video_id, vidName) +".h5", "r")
        logs  = h5py.File(fpath+"log/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        cams  = h5py.File(fpath+"cam/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        attns = h5py.File(fpath+"attn/"+'{}_{}'.format(video_id, vidName) +".h5", "r")

        # (synced) control commands
        timestamp           = np.squeeze(attns["timestamp"][:])
        curvature_value     = np.squeeze(attns["curvature"][:])
        accelerator_value   = np.squeeze(attns["accel"][:])
        speed_value         = np.squeeze(attns["speed"][:])
        course_value        = np.squeeze(attns["course"][:])
        goaldir_value       = np.squeeze(attns["goaldir"][:])
        acc_pred_value      = np.squeeze(attns["pred_accel"][:])
        course_pred_value   = np.squeeze(attns["pred_courses"][:])



        # Will pad +/- 1 second; extract Frames of Interest
        startStamp = timestamp[0] + float((int(sTime)-1))*1000
        endStamp   = timestamp[0] + float((int(eTime)+1))*1000

        ind2interest = np.where(np.logical_and(np.array(timestamp)>=startStamp, np.array(timestamp)<=endStamp))
        if len(ind2interest[0]) > 0:
            samples_with_data += 1
        print('sTime: {}, eTime: {}, sStamp: {}, eStamp: {}, index: {}'.format(sTime, eTime, startStamp, endStamp, len(ind2interest[0])))

        # Extract for each caption the relevant frames/values in the time span including features, attention and logs
        feat         = feats['X'][:]
        feat         = feat[ind2interest]
        attn         = attns['attn'][:]
        attn         = attn[ind2interest]
        context = attns['context'][:]
        context = context[ind2interest]
        img = cams['X'][:]
        img = img[ind2interest]

        speed_value         = speed_value[ind2interest]
        course_value        = course_value[ind2interest]
        accelerator_value   = accelerator_value[ind2interest]
        curvature_value     = curvature_value[ind2interest]
        goaldir_value       = goaldir_value[ind2interest]
        acc_pred_value      = acc_pred_value[ind2interest]
        course_pred_value   = course_pred_value[ind2interest]

        ## feat (for captioning) # Samplen jeden 5. wert
        feat             = feat[::sampleInterval]
        attn             = attn[::sampleInterval]
        context          = context[::sampleInterval]
        img              = img[::sampleInterval]
        speed_value      = speed_value[::sampleInterval]
        course_value     = course_value[::sampleInterval]
        accelerator_value= accelerator_value[::sampleInterval]
        curvature_value  = curvature_value[::sampleInterval]
        goaldir_value    = goaldir_value[::sampleInterval]
        acc_pred_value   = acc_pred_value[::sampleInterval]
        course_pred_value= course_pred_value[::sampleInterval]

        ## padding vectors with zeros from the left until max video sequence ("caption sequence") of 10 is reached
        speed_value       = pad_video(speed_value,       (max_length,))
        course_value      = pad_video(course_value,      (max_length,))
        accelerator_value = pad_video(accelerator_value, (max_length,))
        curvature_value   = pad_video(curvature_value,   (max_length,))
        goaldir_value     = pad_video(goaldir_value,     (max_length,))
        acc_pred_value    = pad_video(acc_pred_value,    (max_length,))
        course_pred_value = pad_video(course_pred_value, (max_length,))
        timestamp         = pad_video(timestamp,         (max_length,))
        mask4Cap          = fill_mask(max_length, feat.shape[0], zero_location='LEFT')
        feat4Cap          = pad_video(feat, (max_length, 64, 12, 20))
        attn4Cap          = pad_video(attn, (max_length, 12*20))
        context4cap       = pad_video(context, (max_length, 64))
        img4cap           = pad_video(img, (max_length, 90, 160, 3))

        # accumulate
        # !!! all_xxx can be indexed for each annotation/caption sequence (for each annotated vehicle action)
        # data format: [num_sequence, vid_frames, value/vector per frame]
        all_feats4Cap[idx]          = feat4Cap
        all_masks4Cap[idx]          = mask4Cap
        all_attns4Cap[idx]          = attn4Cap
        all_contexts4Cap[idx]       = context4cap
        all_imgs4Cap[idx]           = img4cap
        all_logs['timestamp'][idx]  = timestamp
        all_logs['speed'][idx]      = speed_value
        all_logs['course'][idx]     = course_value
        all_logs['accelerator'][idx]= accelerator_value
        all_logs['curvature'][idx]  = curvature_value
        all_logs['goaldir'][idx]    = goaldir_value
        all_logs['pred_accel'][idx] = acc_pred_value
        all_logs['pred_courses'][idx]= course_pred_value

        idx += 1

    print(bcolors.BLUE + '[_build_feat_matrix] max_video_length: {} (caption), {} (control)'.format(max_length, max_length) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] Sample freq: {} Hz'.format(hz/sampleInterval) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] max_log_length: {}'.format(max_length) + bcolors.ENDC)

    return all_feats4Cap, all_masks4Cap, all_logs, all_attns4Cap, all_contexts4Cap, all_imgs4Cap, samples_with_data


def build_feat_matrix_SAX(annotations, max_length, fpath, hz=10, sampleInterval=5, FINETUNING=False):
    print(bcolors.GREEN + '[_build_feat_matrix] Collect feats and masks' + bcolors.ENDC)

    n_examples    = len(annotations) #number of sequences
    max_length_vid= max_length*sampleInterval

    all_logs    = {}
    all_logs['speed']       = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['course']      = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['accelerator'] = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['curvature']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['goaldir']     = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['timestamp']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_accel']  = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_courses']= np.ndarray([n_examples, max_length], dtype=np.float32)


    all_masks4Cap = np.ndarray([n_examples, max_length], dtype=np.float32)
    #all_feats4Cap = np.memmap('/data2/tmp.dat', dtype='float32', mode='w+', shape=(n_examples, max_length, 64, 12, 20))
    all_feats4Cap = np.ndarray([n_examples, max_length, 64, 12, 20], dtype=np.float32)
    all_imgs4Cap = np.ndarray([n_examples, max_length, 90, 160, 3], dtype=np.uint8)
    all_attns4Cap = np.ndarray([n_examples, max_length, 12*20], dtype=np.float32)
    all_contexts4Cap = np.ndarray([n_examples, max_length, 64], dtype=np.float32)

    sTimes       = annotations['sTime']      # staring timestamp
    eTimes       = annotations['eTime']      # ending timestamp
    vidNames     = annotations['vidName']    # video clip name
    video_ids    = annotations['video_id']   # index of video

    idx          = 0
    samples_with_data = 0
    for sTime, eTime, vidName, video_id in zip(sTimes, eTimes, vidNames, video_ids): # For each caption

        print(bcolors.BLUE + '[_build_feat_matrix] vidName: {}'.format(vidName) + bcolors.ENDC)

        # load feats
        #if (os.path.isfile(fpath+"feats_%s/"%(dataset)+'{}_{}'.format(video_id, vidName)+".h5")) == False: continue
        if (os.path.isfile(fpath+"log/"+'{}_{}'.format(video_id, vidName) +".h5")) == False: continue

        feats = h5py.File(fpath+"feat/"+'{}_{}'.format(video_id, vidName) +".h5", "r")
        logs  = h5py.File(fpath+"log/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        cams  = h5py.File(fpath+"cam/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        attns = h5py.File(fpath+"attn/"+'{}_{}'.format(video_id, vidName) +".h5", "r")

        # Check if first or second video part
        if vidName[-1] == "2":
            #Check video ID of part 1
            video_id_p1_1 = "X"
            for file_p1 in sorted(os.listdir(fpath+"attn/")):
                filename_p1 = os.fsdecode(file_p1)
                if filename_p1[-25:] == "_" + vidName[:-1] + "1.h5":
                    video_id_p1 = filename_p1[:2]
                    if video_id_p1[1] == "_":
                        video_id_p1_1 = video_id_p1[0]
                    else:
                        video_id_p1_1 = video_id_p1
                    break

            # Load timestamp from part 1
            if (os.path.isfile(fpath + "attn/" + '{}_{}'.format(video_id_p1_1, vidName[:-1]) + "1.h5")) == False: continue
            attns_p1 = h5py.File(fpath + "attn/" + '{}_{}'.format(video_id_p1_1, vidName[:-1]) + "1.h5", "r")
            #print("Loaded")
            timestamp_p1 = np.squeeze(attns_p1["timestamp"][:])

        # (synced) control commands
        timestamp           = np.squeeze(attns["timestamp"][:])
        curvature_value     = np.squeeze(attns["curvature"][:])
        accelerator_value   = np.squeeze(attns["accel"][:])
        speed_value         = np.squeeze(attns["speed"][:])
        course_value        = np.squeeze(attns["course"][:])
        goaldir_value       = np.squeeze(attns["goaldir"][:])
        acc_pred_value      = np.squeeze(attns["pred_accel"][:])
        course_pred_value   = np.squeeze(attns["pred_courses"][:])

        # Will pad +/- 1 second; extract Frames of Interest
        if vidName[-1] == "2":
            startStamp = timestamp_p1[0] + float((int(sTime) - 3)) * (1000000)  # Difference between SAX and BDDX
            endStamp = timestamp_p1[0] + float((int(eTime) + 3)) * (1000000)
            #print(timestamp_p1[0])
        elif vidName[-1] == "1":
            startStamp = timestamp[0] + float((int(sTime) - 3)) * (1000000)  # Difference between SAX and BDDX
            endStamp = timestamp[0] + float((int(eTime) + 3)) * (1000000)
            #print(timestamp[0])
        else:
            print("Error.")


        ind2interest = np.where(np.logical_and(np.array(timestamp)>=startStamp, np.array(timestamp)<=endStamp))

        if len(ind2interest[0]) > 0:
            samples_with_data += 1
            #all_logs['timestamp'][idx][0][0] = 0.0
        #    continue

        print('sTime: {}, eTime: {}, sStamp: {}, eStamp: {}, index: {}'.format(sTime, eTime, startStamp, endStamp, len(ind2interest[0])))
        #samples_with_data += 1

        # Extract for each caption the relevant frames/values in the time span including features, attention and logs
        feat         = feats['X'][:]
        feat         = feat[ind2interest]
        attn         = attns['attn'][:]
        attn         = attn[ind2interest]
        context = attns['context'][:]
        context = context[ind2interest]
        img = cams['X'][:]
        img = img[ind2interest]

        speed_value         = speed_value[ind2interest]
        course_value        = course_value[ind2interest]
        accelerator_value   = accelerator_value[ind2interest]
        curvature_value     = curvature_value[ind2interest]
        goaldir_value       = goaldir_value[ind2interest]
        acc_pred_value      = acc_pred_value[ind2interest]
        course_pred_value   = course_pred_value[ind2interest]
        timestamp_idx       = timestamp[ind2interest]

        ## feat (for captioning) # Samplen jeden 5. wert
        feat             = feat[::]
        attn             = attn[::]
        context          = context[::]
        img              = img[::]
        speed_value      = speed_value[::]
        course_value     = course_value[::]
        accelerator_value= accelerator_value[::]
        curvature_value  = curvature_value[::]
        goaldir_value    = goaldir_value[::]
        acc_pred_value   = acc_pred_value[::]
        course_pred_value= course_pred_value[::]

        ## padding vectors with zeros from the left until max video sequence ("caption sequence") of 10 is reached
        speed_value       = pad_video(speed_value,       (max_length,))
        course_value      = pad_video(course_value,      (max_length,))
        accelerator_value = pad_video(accelerator_value, (max_length,))
        curvature_value   = pad_video(curvature_value,   (max_length,))
        goaldir_value     = pad_video(goaldir_value,     (max_length,))
        acc_pred_value    = pad_video(acc_pred_value,    (max_length,))
        course_pred_value = pad_video(course_pred_value, (max_length,))
        timestamp         = pad_video(timestamp_idx,         (max_length,))
        mask4Cap          = fill_mask(max_length, feat.shape[0], zero_location='LEFT')
        feat4Cap          = pad_video(feat, (max_length, 64, 12, 20))
        attn4Cap          = pad_video(attn, (max_length, 12*20))
        context4cap       = pad_video(context, (max_length, 64))
        img4cap           = pad_video(img, (max_length, 90, 160, 3))

        # accumulate
        all_feats4Cap[idx]          = feat4Cap
        all_masks4Cap[idx]          = mask4Cap
        all_attns4Cap[idx]          = attn4Cap
        all_contexts4Cap[idx]       = context4cap
        all_imgs4Cap[idx]           = img4cap
        all_logs['timestamp'][idx]  = timestamp
        all_logs['speed'][idx]      = speed_value
        all_logs['course'][idx]     = course_value
        all_logs['accelerator'][idx]= accelerator_value
        all_logs['curvature'][idx]  = curvature_value
        all_logs['goaldir'][idx]    = goaldir_value
        all_logs['pred_accel'][idx] = acc_pred_value
        all_logs['pred_courses'][idx]= course_pred_value

        idx += 1

    print(bcolors.BLUE + '[_build_feat_matrix] max_video_length: {} (caption), {} (control)'.format(max_length, max_length) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] Sample freq: {} Hz'.format(hz/sampleInterval) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] max_log_length: {}'.format(max_length) + bcolors.ENDC)

    return all_feats4Cap, all_masks4Cap, all_logs, all_attns4Cap, all_contexts4Cap, all_imgs4Cap, samples_with_data

def cluster_texts(texts, clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)

    print("Top terms per cluster:")
    order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(clusters):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering

def cluster_annotations(annotations, k=2):
    print(annotations['caption'])

    clusters = cluster_texts(annotations['justification'], k)

    ind_cluster = np.ndarray([len(annotations)], dtype=np.float32)
    for key, index in clusters.items():
    #for key, index in clusters.iteritems():
        ind_cluster[index] = key
        #print(key, index)


    return clusters, ind_cluster

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def tf_count(t, val):
    elements_equal_to_value = tf.math.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def tf_is_once(t, val):
    counts = tf_count(t, val)
    is_once = tf.math.logical_not(tf.math.equal(counts, 1))
    as_ints = tf.cast(is_once, tf.int32)
    return as_ints #1 if more or less than once or 0 if once







