# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import json
import config_
import lstm
import bilstm
import lstmCRF
import bilstmCRF
import argparse


def save_label(flag, file_to_save_model, data_dir, data_name, batch_size=100):
    posdt_dir = os.path.join(data_dir+'posdt.json')
    with open(posdt_dir,'r') as f:
        posdt = json.load(f)
    inv_map = dict(zip(posdt.values(), posdt.keys())) 

    config = config_.Config()
    if flag == '1':
    	tagger = lstm.Tagger(config=config)
    elif flag == '2':
    	tagger = bilstm.biRNNTagger(config=config)
    elif flag =='3':
        tagger = lstmCRF.CRFTagger(config=config)
    elif flag =='4':
    	tagger = bilstmCRF.BCRFTagger(config=config)
    else:
    	print("No such model")
    	return

    init = tf.global_variables_initializer()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(data_dir,file_to_save_model)))

        f = open(os.path.join(data_dir, file_to_save_model,'pred_'+data_name.strip('.json')+'.txt'),'w',encoding='utf-8')
        f.write('true'+'\n'+'prediction'+'\n'+'\n')
        
        X_train, y_train, data_seg, label_seg = lstm.readdata(os.path.join(data_dir,data_name))
        [x_batch, y_batch, data_seg_batch, label_seg_batch] = lstm.multiminibatch([X_train, y_train, data_seg, label_seg], batch_size)
        acc=0
        for step in range(len(x_batch)):
            batch_xs = x_batch[step]
            batch_ys = y_batch[step]
            batch_data_seg = data_seg_batch[step]
            batch_label_seg = label_seg_batch[step]
            traininputs, trainlabels, trainseq_length = tagger.pre_process(batch_xs, batch_ys)
            pred = sess.run(tagger.prediction,feed_dict={tagger.x: traininputs, tagger.length: trainseq_length, tagger.dropout: 0.0})
            acc += sess.run(tagger.accuracy,feed_dict={tagger.x: traininputs, tagger.y: trainlabels, tagger.length: trainseq_length, tagger.dropout: 0.0})
            if flag == '1' or flag == '2':
                tag_ls = np.argmax(pred,axis=1) #if use model LSTM/BLSTM, pred return a onehot encoding vector
            elif flag == '3' or flag == '4':
                tag_ls = pred #if use model combine CRF, pred return a label
            
            ix=0
            for count in range(len(trainseq_length)):

                sen = batch_data_seg[count]
                sentence = sen.split(' ')#[:-1]
                lab = batch_label_seg[count]
                lals = lab.split(' ')#[:-1]
                true=''
                #print(sentence)
                for word,tag in zip(sentence,lals):
                    true = true+word+'_'+tag+' '
                
                f.write(true+'\n')
                result=''
                for i in range(ix, ix+trainseq_length[count]):
                    result = result + sentence[i-ix] + '_' + inv_map[tag_ls[i]] + ' '              
                f.write(result+'\n')
                f.write('\n')
                ix=ix+trainseq_length[count]

            f.flush()
            
        acc=acc/len(x_batch)
        f.write('accuracy: '+str(acc))
        print('accuracy: %f' %acc)
        f.close()
        print(data_name+' saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", type=str, default='../data/', dest="data_dir", help="specify your data directory")
    parser.add_argument("--model_path", type=str, default='lstmcrf', dest="file_to_save_model",help="name the file to save your model")
    parser.add_argument("--model_type", type=str, default='3', dest="flag", help="the model to be trained \n1: LSTM \n2: BiLSTM \n3: LSTM+CRF \n4: BLSTM+CRF \n")
    parser.add_argument("--data_name", type=str, default='testing_dt.json', help="file to be put into POS tagger to see prediction")
    arg=parser.parse_args()
    data_dir = arg.data_dir
    file_to_save_model = arg.file_to_save_model
    flag = arg.flag
    data_name = arg.data_name
    save_label(flag, file_to_save_model, data_dir, data_name)
