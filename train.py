from dataclasses import asdict

import sys

import argparse
import logging
import wandb as wandb
from typing import List, Dict, Set, Tuple

import torch
import os
import json
import numpy as np
import random

from itertools import chain

from expred import metrics
from expred.params import MTLParams
from expred.models.mlp_mtl import BertMTL, BertClassifier
from expred.tokenizer import BertTokenizerWithMapping
from expred.models.pipeline.mtl_pipeline_utils import decode
from expred.utils import load_datasets, load_documents, write_jsonl, Annotation
from expred.models.pipeline.mtl_token_identifier import train_mtl_token_identifier
from expred.models.pipeline.mtl_evidence_classifier import train_mtl_evidence_classifier
from expred.eraser_utils import get_docids

BATCH_FIRST = True
torch.cuda.empty_cache()


#-----Defining methods for separating the MultiRc Train dataset into 3 sets: Train, val and test-----#
import math
import random
import json
def count_no_lines(filename):
    with open(filename, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    print('Total Number of lines:', count + 1)
    return count+1

def create_datapoints(new_file_name,file_name,indices):
    with open(new_file_name,'w') as fp:
        with open(file_name,'r') as fp1:
            for count, line in enumerate(fp1):
                json_object = json.loads(line)
                #print(type(json_object))
                if count+1 in indices:
                    json.dump(json_object, fp)
                    fp.write('\n')

#-----------------------end of separating part------------------------------------#



def initialize_models(conf: dict,
                      tokenizer: BertTokenizerWithMapping,
                      batch_first: bool) -> Tuple[BertMTL, BertClassifier, Dict[int, str]]:
    """
    Does several things:
    1. Create a mapping from label names to ids
    2. Configure and create the multi task learner, the first stage of the model (BertMTL)
    3. Configure and create the evidence classifier, second stage of the model (BertClassifier)
    :param conf:
    :param tokenizer:
    :param batch_first:
    :return: BertMTL, BertClassifier, label mapping
    """
    assert batch_first
    max_length = conf['max_length']
    # label mapping
    labels = dict((y, x) for (x, y) in enumerate(conf['classes']))
    

    # configure multi task learner
    mtl_params = MTLParams
    mtl_params.num_labels = len(labels)
    mtl_params.dim_exp_gru = conf['dim_exp_gru']
    mtl_params.dim_cls_linear = conf['dim_cls_linear']
    bert_dir = conf['bert_dir']
    use_half_precision = bool(conf['mtl_token_identifier'].get('use_half_precision', 1))
    evidence_identifier = BertMTL(bert_dir=bert_dir,
                                  tokenizer=tokenizer,
                                  mtl_params=mtl_params,
                                  max_length=max_length,
                                  use_half_precision=use_half_precision)

    #set up the evidence classifier
    use_half_precision  =  bool(conf['evidence_classifier'].get('use_half_precision', 1))
    print("Labels ",labels)
    print("use half precision",use_half_precision)
    evidence_classifier = BertClassifier(bert_dir=bert_dir,
                                         pad_token_id=tokenizer.pad_token_id,
                                         cls_token_id=tokenizer.cls_token_id,
                                         sep_token_id=tokenizer.sep_token_id,
                                         num_labels=mtl_params.num_labels,
                                         max_length=max_length,
                                         mtl_params=mtl_params,
                                         use_half_precision=use_half_precision)

    return evidence_identifier, evidence_classifier, labels


logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# or, uncomment the following sentences to make it more than random
# rand_seed_1 = ord(os.urandom(1)) * ord(os.urandom(1))
# rand_seed_2 = ord(os.urandom(1)) * ord(os.urandom(1))
# rand_seed_3 = ord(os.urandom(1)) * ord(os.urandom(1))
# random.seed(rand_seed_1)
# np.random.seed(rand_seed_2)
# torch.manual_seed(rand_seed_3)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

import time
from time import gmtime, strftime

def main():
    # setup the Argument Parser
    parser = argparse.ArgumentParser(description=('Trains a pipeline model.\n'
                                                  '\n'
                                                  'Step 1 is evidence identification, the MTL happens here. It '
                                                  'predicts the label of the current sentence and tags its\n '
                                                  '        sub-tokens in the same time \n'
                                                  '    Step 2 is evidence classification, a BERT classifier takes the output of the evidence identifier and predicts its \n'
                                                  '        sentiment. Unlike in Deyong et al. this classifier takes in the same length as the identifier\'s input but with \n'
                                                  '        irrational sub-tokens masked.\n'
                                                  '\n'
                                                  '    These models should be separated into two separate steps, but at the moment:\n'
                                                  '    * prep data (load, intern documents, load json)\n'
                                                  '    * convert data for evidence identification - in the case of training data we take all the positives and sample some negatives\n'
                                                  '        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a broader sampling of negative values.\n'
                                                  '    * train evidence identification\n'
                                                  '    * convert data for evidence classification - take all rationales + decisions and use this as input\n'
                                                  '    * train evidence classification\n'
                                                  '    * decode first the evidence, then run classification for each split\n'
                                                  '\n'
                                                  '    '), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--conf', dest='conf', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument('--batch_size', type=int, required=False, default=None,
                        help='Overrides the batch_size given in the config file. Helpful for debugging')
    args = parser.parse_args(args=['--data_dir', 'data/multirc', '--output_dir', './trained_data','--conf','params/multirc_expred.json'])
    print(args.data_dir)

    #wandb.init(project="expred")
    #wandb.init(project="expred")
    #Configure
    os.makedirs(args.output_dir, exist_ok=True)

    #loads the config
    
    with open(args.conf, 'r') as fp:
        logger.info(f'Loading configuration from {args.conf}')
        conf = json.load(fp)
        if args.batch_size is not None:
            logger.info(
                'Overwriting batch_sizes'
                f'(mtl_token_identifier:{conf["mtl_token_identifier"]["batch_size"]}'
                f'evidence_classifier:{conf["evidence_classifier"]["batch_size"]})'
                f'provided in config by command line argument({args.batch_size})'
            )
            conf['mtl_token_identifier']['batch_size'] = args.batch_size
            conf['evidence_classifier']['batch_size'] = args.batch_size
        logger.info(f'Configuration: {json.dumps(conf, indent=2, sort_keys=True)}')
    

    #todo add seeds
    #wandb.config.update(conf)
    #wandb.config.update(args)

    #load the annotation data

    #here train, val and test are the list of Annotation dataclass
    
    lines=count_no_lines("/home/mt1/21CS60R28/expredAI2/data/multirc/train.jsonl")
    main_indices=list()
    for i in range(lines+1):
        main_indices.append(i+1)

    
    train_size=int(math.ceil((lines)*(0.8)))
    train_indices=random.sample(main_indices,train_size)
    print(len(train_indices))
    print(train_size)

    create_datapoints("/home/mt1/21CS60R28/expredAI2/data/multirc/train1.jsonl","/home/mt1/21CS60R28/expredAI2/data/multirc/train.jsonl",train_indices)

    val_test_indices=list()
    for index in main_indices:
        if index not in train_indices:
            val_test_indices.append(index)

    val_size=int(math.ceil(len(val_test_indices)))
    val_indices=random.sample(val_test_indices,val_size)
    print(val_size)
    print(len(val_indices))


    create_datapoints("/home/mt1/21CS60R28/expredAI2/data/multirc/val1.jsonl","/home/mt1/21CS60R28/expredAI3/data/multirc/train.jsonl",val_indices)

    
     #------END OF SEPARATION PART------#
    
   

   
    
    train, val, test = load_datasets(args.data_dir)
    count_true_train=0
    count_false_train=0
    count_false_val=0
    count_true_val=0
    count_true_test=0
    count_false_test=0
    for i in range(len(train)):
        if train[i].classification=="True":
            count_true_train=count_true_train+1
        else:
             count_false_train=count_false_train+1
    print("True Examples :",count_true_train)
    print("False Examples :",count_false_train)
    print("Total Examples :",len(train))


    for i in range(len(val)):
         if val[i].classification=="True":
            count_true_val=count_true_val+1
         else:
             count_false_val=count_false_val+1
    print("True Examples :",count_true_val)
    print("False Examples :",count_false_val)
    print("Total Examples :",len(val))

    for i in range(len(test)):
        if test[i].classification=="True":
             count_true_test=count_true_test+1
        else:
            count_false_test=count_false_test+1
    print("True Examples :",count_true_test)
    print("False Examples :",count_false_test)
    print("Total Examples :",len(test))
 
    
    # print()
    # print("Printing one Trained Data Example----")
    # for i in range(len(train)):
    #     print("Annotation :",train[i].annotation_id)
    #     print("Query :",train[i].query)
    #     print("Evidences :")
    #     for j in train[i].evidences:
    #         print(j)
    #     print("Classification :",train[i].classification)
    #     print("DocIDs :",train[i].docids)
    #     break
    # print()
    # print("Printing one Validation Data Example----")
    # for i in range(len(val)):
    #     print("Annotation :",train[i].annotation_id)
    #     print("Query :",val[i].query)
    #     print("Evidences :")
    #     for j in val[i].evidences:
    #         print(j)
    #     print("Classification :",val[i].classification)
    #     print("DocIDs :",val[i].docids)
    #     break
    
    # print()
    # print("Printing one Test Data Example----")
    # for i in range(len(test)):
    #     print("Annotation :",train[i].annotation_id)
    #     print("Query :",test[i].query)
    #     print("Evidences :")
    #     for j in test[i].evidences:
    #         print(j)
    #     print("Classification :",test[i].classification)
    #     print("DocIDs :",test[i].docids)
    #     break

    
    
    #get's all docids in which evidences of train set, val set and test set is present.
    docids: Set[str] = set(chain.from_iterable(map(lambda ann: get_docids(ann),
                                               chain(train, val, test))))
    



    #Printing all docIDS in which evidences of train set, val set and test set is present.
    # print("\nDoc IDs: ",docids)
    # print()

    

    #All the Documents related to above docIDs gets loaded from ./docs Folder
    #'documents' contains list of tokenized words
    documents: Dict[str, List[List[str]]] = load_documents(args.data_dir, docids)
    
    # print("\nPrinting one Document ----")
    # for key in documents.keys():
    #     print("Document ID:",key)
    #     for ele in documents[key]:
    #         print(ele)
    #     break
    
    
    logger.info(f'Load {len(documents)} documents')
    #this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    #logger.info(f'We have {len(word_interner)} wordpieces')

    #tokenizes and caches tokenized_docs, same for annotations
    #todo typo here? slides == slices (words?)
    #converts 
    
    #Calling BERT Tokenizer(bert-based-uncased) for converting tokens to its corresponding IDs.
    tokenizer = BertTokenizerWithMapping.from_pretrained(conf['bert_vocab'])

    #Converting list of Tokenized words of documents into a list of token IDs. 
    tokenized_docs, tokenized_doc_token_slices = tokenizer.encode_docs(documents, args.output_dir)
     
    # print("\nPrinting one Tokenized Document ----")
    # for key in tokenized_docs.keys():
    #     print("Tokenized Document ID:",key)
    #     for ele in tokenized_docs[key]:
    #         print(ele)
    #     break
    # print()
    

    #tokenized_doc_token_slices is a dictionary of list-of-tuple token mapping 
    # print("\nPrinting Tokenized Document Token Slices ----")
    # for key in tokenized_doc_token_slices.keys():
    #     print("Tokenized Document ID:",key)
    #     for ele in tokenized_doc_token_slices[key]:
    #         print(ele)
    #     break
    # print()

    
    #Converting Annotations of train,validation and test set into list of token IDs
    indexed_train, indexed_val, indexed_test = [tokenizer.encode_annotations(data) for data in [train, val, test]]
    
    # print("Indexed Trained ------ ")
    # print("Printing one Trained Data Example----")
    # for i in range(len(indexed_train)):
    #     print("Annotation :",indexed_train[i].annotation_id)
    #     print("Query :",indexed_train[i].query)
    #     print("Evidences :")
    #     for j in indexed_train[i].evidences:
    #         print(j)
    #     print("Classification :",indexed_train[i].classification)
    #     print("DocIDs :",indexed_train[i].docids)
    #     break


    # print("Document without Tokenization :")
    # for ele in documents['Fiction-stories-masc-A_Wasted_Day-2.txt']:
    #         print(ele)

    

    # print("Document with Tokenization :")
    # for ele in tokenized_docs['Fiction-stories-masc-A_Wasted_Day-2.txt']:
    #         print(ele)

    
    #Setting up the Parameters for MTL token identifier and Evidence Classifier
    mtl_token_identifier, evidence_classifier, labels_mapping =initialize_models(conf, tokenizer, batch_first=BATCH_FIRST)
    
    logger.info('Beginning training of the MTL identifier')
     
    
    from datetime import datetime
    start = datetime.now()
    print("INITIAL TIME for mtl token identifier classifier------",start)
    mtl_token_identifier = mtl_token_identifier.cuda()
    mtl_token_identifier, mtl_token_identifier_results, \
    train_machine_annotated, eval_machine_annotated, test_machine_annotated = \
        train_mtl_token_identifier(mtl_token_identifier,
                                   args.output_dir,
                                   indexed_train,
                                   indexed_val,
                                   indexed_test,
                                   labels_mapping=labels_mapping,
                                   interned_documents=tokenized_docs,
                                   source_documents=documents,
                                   token_mapping=tokenized_doc_token_slices,
                                   model_pars=conf,
                                   tensorize_model_inputs=True)
    
    
    
    
    
    # Machine annotated contains three things for each Data Points-
       #1. Evidence Data
          #it contains two things-
              # 1. Classification ie either True Or False
              # 2. Sentence Evidence which contains-
                #  kls (ORIGINAL RATIONALES)
                #  ann_id
                #  query
                #  doc_id
                #  index
                #  sentence
                #  has_evidence
       #2. Soft rationale prediction for tokens
       #3. Hard rationale prediction for tokens.
    
    
    #print("Train Machine Annotated :")
    #print(len(train_machine_annotated[0][0][1].kls))
    #print(len(train_machine_annotated[0][2]))
    
    
    #Making all the tokens as rationales for the second model
    for i in range(len(train_machine_annotated)):
        for j in range(len(train_machine_annotated[i][2])):
            train_machine_annotated[i][2][j]=1
    

    for i in range(len(eval_machine_annotated)):
        for j in range(len(eval_machine_annotated[i][2])):
            eval_machine_annotated[i][2][j]=1
    

    for i in range(len(test_machine_annotated)):
        for j in range(len(test_machine_annotated[i][2])):
            test_machine_annotated[i][2][j]=1

    

    
    # print("Train Samples for BERT classifier -")
    # for i in range(len(train_machine_annotated)):
    #     print("Sentence No :",i+1)
    #     print("Classification :",train_machine_annotated[i][0][0])
    #     print("Query :",train_machine_annotated[i][0][1].query)
    #     print("Sentences :",train_machine_annotated[i][0][1].sentence)
    #     print()


    # print()
    # print("Validation Samples for BERT classifier -")
    # for i in range(len(eval_machine_annotated)):
    #     print("Sentence No :",i+1)
    #     print("Classification :",eval_machine_annotated[i][0][0])
    #     print("Query :",eval_machine_annotated[i][0][1].query)
    #     print("Sentences :",eval_machine_annotated[i][0][1].sentence)
    #     print()
    
    # print()


    #print("Train machine Annotated---")
    #print(train_machine_annotated)
    #print("MTL_Token Identifier Results---")
    #print(mtl_token_identifier_results)
    
    torch.cuda.empty_cache()
    mtl_token_identifier = mtl_token_identifier.cpu()
    end=datetime.now()
    print("END TIME for mtl token identifier classifier-----",end) 
    print("Time taken by mtl token identifier :",end-start)   
    #evidence identifier ends
    
    logger.info('Beginning training of the evidence classifier')
    evidence_classifier = evidence_classifier.cuda()
    optimizer = None
    scheduler = None
    #trains the classifier on the masked (based on rationales) documents
    start = datetime.now()
    print("START TIME FOR EVIDENCE CLASSIFIER :",start)
    print()
    evidence_classifier, evidence_class_results = train_mtl_evidence_classifier(evidence_classifier,
                                                                                args.output_dir,
                                                                                train_machine_annotated,
                                                                                eval_machine_annotated,
                                                                                tokenized_docs,
                                                                                conf,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                class_interner=labels_mapping,
                                                                                tensorize_model_inputs=True)
    #evidence classifier ends
    end = datetime.now()

    print()
    print()
    print("END TIME FOR EVIDENCE CLASSIFIER :",end)
    print("Time Taken by the evidence_classifier-----",end-start)

    
    
    
    logger.info('Beginning final decoding')
    mtl_token_identifier = mtl_token_identifier.cuda()
    
    pipeline_batch_size = min([conf['evidence_classifier']['batch_size'], conf['mtl_token_identifier']['batch_size']])
    pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier=mtl_token_identifier,
                                                                        evidence_classifier=evidence_classifier,
                                                                        train=indexed_train,
                                                                        mrs_train=train_machine_annotated,
                                                                        val=indexed_val,
                                                                        mrs_eval=eval_machine_annotated,
                                                                        test=indexed_test,
                                                                        mrs_test=test_machine_annotated,
                                                                        source_documents=documents,
                                                                        interned_documents=tokenized_docs,
                                                                        token_mapping=tokenized_doc_token_slices,
                                                                        class_interner=labels_mapping,
                                                                        tensorize_modelinputs=True,
                                                                        batch_size=pipeline_batch_size,
                                                                        tokenizer=tokenizer)

    
    write_jsonl(train_decoded, os.path.join(args.output_dir, 'train_decoded.jsonl'))
    write_jsonl(val_decoded, os.path.join(args.output_dir, 'val_decoded.jsonl'))
    write_jsonl(test_decoded, os.path.join(args.output_dir, 'test_decoded.jsonl'))
    with open(os.path.join(args.output_dir, 'identifier_results.json'), 'w') as ident_output, \
            open(os.path.join(args.output_dir, 'classifier_results.json'), 'w') as class_output:
        ident_output.write(json.dumps(mtl_token_identifier_results))
        class_output.write(json.dumps(evidence_class_results))
    for k, v in pipeline_results.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                logging.info(f'Pipeline results for {k}, {k1}={v1}')
        else:
            logging.info(f'Pipeline results {k}\t={v}')
    
    
    #decode ends
    scores = metrics.main(
        [
            '--data_dir', args.data_dir,
            '--split', 'test',
            '--results', os.path.join(args.output_dir, 'test_decoded.jsonl'),
            '--score_file', os.path.join(args.output_dir, 'test_scores.jsonl')
        ]
    )
    
    #wandb.log(scores)

    #wandb.save(os.path.join(args.output_dir, '*.jsonl'))
    #torch.cuda.empty_cache()
    
    
    
    



if __name__ == '__main__':
    main()
