import torch
import os
import logging
import numpy as np
import random
import wandb

from typing import List, Dict, Tuple, Callable, Union, Any
from collections import OrderedDict, namedtuple, defaultdict
from sklearn.metrics import accuracy_score, classification_report
from itertools import chain
from torch import nn
from torch.nn import Module

from expred.eraser_utils import chain_sentence_evidences
from expred.models.model_utils import PaddedSequence
from expred.utils import Annotation
from expred.models.pipeline.pipeline_utils import (
    SentenceEvidence, score_token_rationales)
from expred.models.pipeline.mtl_pipeline_utils import (
    annotations_to_mtl_token_identification,
    make_mtl_token_preds_epoch
)
from expred.models.losses import resampling_rebalanced_crossentropy

AnnotatedDocument = namedtuple('AnnotatedDocument', 'kls evd ann_id query docid index sentence')


def _get_sampling_method(params):
    if params['sampling_method'] == 'whole_document':
        def whole_document_sampler(sentences, _):
            return chain_sentence_evidences(sentences)

        return whole_document_sampler
    else:
        raise NotImplementedError


def _prep_data_for_epoch(evidence_data: Tuple[str, Dict[str, Dict[str, List[SentenceEvidence]]]],
                         sampler: Callable[
                             [List[SentenceEvidence], Dict[str, List[SentenceEvidence]]], List[SentenceEvidence]]
                         ) -> List[SentenceEvidence]:
    """
    Shuffle the annotations and sample from documents (can also be the whole document depending on the sampler)
    :param evidence_data:
    :param sampler:
    :return:
    """
    output_annotations = []
    ann_ids = sorted(evidence_data.keys())
    # in place shuffle so we get a different per-epoch ordering
    random.shuffle(ann_ids)
    for ann_id in ann_ids:
        for docid, sentences in evidence_data[ann_id][1].items():
            data = sampler(sentences, None)
            #evidence_data[ann_id][0] is the Classification
            #data contains kls, ann_id,sentence,doc_id,index,query,has evidence
            output_annotations.append((evidence_data[ann_id][0], data))  

    return output_annotations


def train_mtl_token_identifier(mtl_token_identifier: nn.Module,
                               save_dir: str,
                               train: List[Annotation],
                               val: List[Annotation],
                               test: List[Annotation],
                               interned_documents: Dict[str, List[List[int]]],
                               source_documents: Dict[str, List[List[str]]],
                               token_mapping: Dict[str, List[List[Tuple[int, int]]]],
                               model_pars: dict,
                               labels_mapping: Dict[str, int],
                               optimizer=None,
                               scheduler=None,
                               tensorize_model_inputs: bool = True) -> Tuple[
    Module, Union[Dict[str, list], Any], List[Tuple[Union[SentenceEvidence, List[SentenceEvidence]], Any, Any]], List[
        Tuple[Union[SentenceEvidence, List[SentenceEvidence]], Any, Any]], List[
        Tuple[Union[SentenceEvidence, List[SentenceEvidence]], Any, Any]]]:
    """Trains a module for token-level rationale identification.
    This method tracks loss on the entire validation set, saves intermediate
    models, and supports restoring from an unfinished state. The best model on
    the validation set is maintained, and the model stops training if a patience
    (see below) number of epochs with no improvement is exceeded.
    As there are likely too many negative examples to reasonably train a
    classifier on everything, every epoch we subsample the negatives.

    :param mtl_token_identifier: token-wise evidence identifier using Multi-Task Learning
    :param save_dir: a place to save intermediate and final results and models.
    :param train: a List of interned Annotation objects.
    :param val: a List of interned Annotation objects.
    :param test: a List of interned Annotation objects.
    :param interned_documents: a Dict of interned sentences
    :param source_documents: a Dict of original sentences, used for sub-token alignment
    :param token_mapping: a mapping from original token to sub tokens
    :param model_pars: model parameters
    :param labels_mapping: a mapping from str labels to their int ids
    :param optimizer: what pytorch optimizer to use, if none, initialize Adam
    :param scheduler: optional, do we want a scheduler involved in learning?
    :param tensorize_model_inputs: should we convert our data to tensors before passing it to the model?
                                   Useful if we have a model that performs its own tokenization (e.g. BERT as a Service)
    :returns
        the trained MTL evidence token identifier
        the intermediate results
        machine-annotated train/eval/test datasets
    """



    # set up output folder structure
    logging.info(f'Beginning training with {len(train)} annotations, {len(val)} for validation')
    evidence_identifier_output_dir = os.path.join(save_dir, 'evidence_token_identifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_identifier_output_dir, exist_ok=True)

    model_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_token_identifier.pt')
    epoch_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_token_identifier_epoch_data.pt')

    # set up training (optimizer, loss (both), sampling_method, ...)
    if optimizer is None:
        optimizer = torch.optim.Adam(mtl_token_identifier.parameters(), lr=model_pars['mtl_token_identifier']['lr'])
    cls_criterion = nn.BCELoss(reduction='none')

    exp_criterion = resampling_rebalanced_crossentropy(seq_reduction='none')  # nn.CrossEntropyLoss(reduction='none')
    sampling_method = _get_sampling_method(model_pars['mtl_token_identifier'])
    batch_size = model_pars['mtl_token_identifier']['batch_size']
    max_length = model_pars['max_length']
    epochs = model_pars['mtl_token_identifier']['epochs']

    patience = model_pars['mtl_token_identifier']['patience']
    max_grad_norm = model_pars['mtl_token_identifier'].get('max_grad_norm', None)
    par_lambda = model_pars['mtl_token_identifier']['par_lambda']
    # annotation id -> docid -> [SentenceEvidence])
    # calculates the classification of the sequence tokens and some other stuff
    # source documents contain actual documents while intered documents contains tokenized documents
    evidence_train_data = annotations_to_mtl_token_identification(train,
                                                                  source_documents=source_documents,
                                                                  interned_documents=interned_documents,
                                                                  token_mapping=token_mapping)
    evidence_val_data = annotations_to_mtl_token_identification(val,
                                                                source_documents=source_documents,
                                                                interned_documents=interned_documents,
                                                                token_mapping=token_mapping)

    evidence_test_data = annotations_to_mtl_token_identification(test,
                                                                 source_documents=source_documents,
                                                                 interned_documents=interned_documents,
                                                                 token_mapping=token_mapping)

    device = next(mtl_token_identifier.parameters()).device
    
    
    
    #Each of the above evidence contains two things for each of ann_is present in the data points of dataset-
    # 1. Classification ie either True Or False
    # 2. Sentence Evidence which contains-
    #    kls
    #    ann_id
    #    query
    #    doc_id
    #    index
    #    sentence
    #    has_evidence
    # print("\nEvidence Train Data-")
    # print(evidence_train_data.keys())
    # for ele in evidence_train_data['Fiction-stories-masc-A_Wasted_Day-2.txt:0:0'][1]['Fiction-stories-masc-A_Wasted_Day-2.txt']:
    #     print(ele)
    results = {
        'sampled_epoch_train_losses': [],
        'epoch_val_total_losses': [],
        'epoch_val_cls_losses': [],
        'epoch_val_exp_losses': [],
        'epoch_val_exp_acc': [],
        'epoch_val_exp_f': [],
        'epoch_val_cls_acc': [],
        'epoch_val_cls_f': [],
        'full_epoch_val_rationale_scores': []
    }

    # allow restoring an existing training run
    start_epoch = 0
    best_epoch = -1
    best_val_total_loss = float('inf')
    best_model_state_dict = None
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        mtl_token_identifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        #print(epoch_data.get('done',0))
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        #if bool(epoch_data.get('done', 0)):
            #start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in mtl_token_identifier.state_dict().items()})
    logging.info(f'Training evidence identifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = _prep_data_for_epoch(evidence_train_data, sampling_method)
        epoch_val_data = _prep_data_for_epoch(evidence_val_data, sampling_method)
        sampled_epoch_train_loss = 0
        losses = defaultdict(lambda: [])
        mtl_token_identifier.train()
        logging.info(f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        # one epoch of training
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            # we sample every time to thereoretically get a better representation of instances over the corpus.
            # this might just take more time than doing so in advance.
            labels, targets, queries, sentences, has_evidence = zip(
                *[(s[0], s[1].kls, s[1].query, s[1].sentence, s[1].has_evidence) for s in batch_elements])

            # one hot encoding for classification
            # print("Labels :")
            # print(labels)

            # print("Targets :")
            # print(targets)

            # print("Queries :")
            # print(queries)

            # print("Sentences :")
            # print(sentences)

            # print("Has Evidences :")
            # print(has_evidence)
            # print("Label mapping :",labels_mapping)
            labels = [[i == labels_mapping[label] for i in range(len(labels_mapping))] for label in labels]
            # print("Labels :",labels)
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            # print("One Hot Representation Labels :")
            # print(labels)

            ids = [(s[1].ann_id, s[1].docid, s[1].index) for s in batch_elements]

            # truncation

            #Combining queries length with targets length along with cls and sep token
            cropped_targets = [[0] * (len(query) + 2)  # length of query and overheads such as [cls] and [sep]
                               + list(target[:(max_length - len(query) - 2)]) for query, target in
                               zip(queries, targets)]
            
            # print("Cropped Targets :")
            # print(cropped_targets)
            # print(len(targets[0]))
            # print(len(cropped_targets[0]))
            cropped_targets = PaddedSequence.autopad(
                [torch.tensor(t, dtype=torch.float, device=device) for t in cropped_targets],
                batch_first=True, device=device)
            
            # print("Cropped Targets with Tensors :")
            # print(cropped_targets)

            targets = [[0] * (len(query) + 2)  # length of query and overheads such as [cls] and [sep]
                       + list(target) for query, target in zip(queries, targets)]
            targets = PaddedSequence.autopad([torch.tensor(t, dtype=torch.float, device='cpu') for t in targets],
                                             batch_first=True, device='cpu')
            
            # print("Targets with Tensors :")
            # print(targets) 


            #Coverting the queries ans sentence into tensored valued input
            if tensorize_model_inputs:
                if all(q is None for q in queries):
                    queries = [torch.tensor([], dtype=torch.long) for _ in queries]
                else:
                    assert all(q is not None for q in queries)
                    queries = [torch.tensor(q, dtype=torch.long) for q in queries]
                sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]

            # prediction
            # print("Tensorized Query and Sentences :")
            # print(queries)
            # print(sentences)


            preds = mtl_token_identifier(queries, ids, sentences)
            cls_preds, exp_preds, attention_masks = preds

            # print("Exp Predictions----")
            # print(exp_preds.size())
            # print("Predictions :")
            # print(cls_preds)
            # print("Actual Output :")
            # print(labels)
            cls_loss = cls_criterion(cls_preds, labels).mean(dim=-1).sum()

            exp_loss_per_instance = exp_criterion(exp_preds, cropped_targets.data.squeeze()).mean(dim=-1)
            has_evidence_mask = torch.tensor(has_evidence, dtype=float, device=exp_loss_per_instance.device)

            exp_loss = (exp_loss_per_instance * has_evidence_mask).sum()
            loss = cls_loss + par_lambda * exp_loss

            losses['cls_loss'].append(cls_loss.item())
            losses['exp_loss'].append(exp_loss.item())
            losses['loss'].append(loss.item())

            sampled_epoch_train_loss += loss.item()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(mtl_token_identifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        sampled_epoch_train_loss /= len(epoch_train_data)
        print()
        results['sampled_epoch_train_losses'].append(sampled_epoch_train_loss)
        print()

        mean_losses = {f'train_{key}':np.mean(loss) for key, loss in losses.items()}
        mean_losses['epoch'] = epoch
        #wandb.log(mean_losses)
        logging.info(f'Epoch {epoch} training loss {sampled_epoch_train_loss}')

        # validation
        with torch.no_grad():
            mtl_token_identifier.eval()
            epoch_val_total_loss, epoch_val_cls_loss, epoch_val_exp_loss, \
            epoch_val_soft_pred, epoch_val_hard_pred, epoch_val_token_targets, \
            epoch_val_pred_labels, epoch_val_labels = \
                make_mtl_token_preds_epoch(mtl_token_identifier,
                                           epoch_val_data,
                                           labels_mapping,
                                           token_mapping,
                                           batch_size,
                                           max_length,
                                           par_lambda,
                                           device,
                                           cls_criterion,
                                           exp_criterion,
                                           tensorize_model_inputs)
            # epoch_val_soft_pred = list(chain.from_iterable(epoch_val_soft_pred.tolist()))
            # epoch_val_hard_pred = list(chain.from_iterable(epoch_val_hard_pred))
            # epoch_val_truth = list(chain.from_iterable(epoch_val_truth))
            results['epoch_val_total_losses'].append(epoch_val_total_loss)
            results['epoch_val_cls_losses'].append(epoch_val_cls_loss)
            results['epoch_val_exp_losses'].append(epoch_val_exp_loss)
            epoch_val_hard_pred_chained = list(chain.from_iterable(epoch_val_hard_pred))
            epoch_val_token_targets_chained = list(chain.from_iterable(epoch_val_token_targets))
            results['epoch_val_exp_acc'].append(accuracy_score(epoch_val_token_targets_chained,
                                                               epoch_val_hard_pred_chained))
            results['epoch_val_exp_f'].append(classification_report(epoch_val_token_targets_chained,
                                                                    epoch_val_hard_pred_chained,
                                                                    labels=[0, 1],  # of course rational and irrational
                                                                    output_dict=True))
            
            # print("Actual and predicted value of Validation in training :")
            # print("Predictions :",epoch_val_pred_labels)
            # print("Actual :",epoch_val_labels)
            
            flattened_epoch_val_pred_labels = [np.argmax(x) for x in epoch_val_pred_labels]
            flattened_epoch_val_labels = [np.argmax(x) for x in epoch_val_labels]
            # print(flattened_epoch_val_pred_labels)
            # print(flattened_epoch_val_labels)
            results['epoch_val_cls_acc'].append(accuracy_score(flattened_epoch_val_pred_labels,
                                                               flattened_epoch_val_labels))
            
            results['epoch_val_cls_f'].append(classification_report(flattened_epoch_val_labels,
                                                                    flattened_epoch_val_pred_labels,
                                                                    labels=[v for _, v in labels_mapping.items()],
                                                                    output_dict=True))
            results['full_epoch_val_rationale_scores'].append(
                score_token_rationales(val, source_documents,
                                       epoch_val_data,
                                       token_mapping,
                                       epoch_val_hard_pred,
                                       epoch_val_soft_pred))

            validation_metrics = {metric:values[-1] for metric, values in results.items()}
            validation_metrics['epoch'] = epoch
            # epoch_val_soft_pred_for_scoring = [[[1 - z, z] for z in y] for y in epoch_val_soft_pred]
            # logging.info(
            #    f'Epoch {epoch} full val loss {epoch_val_total_loss}, accuracy: {results["epoch_val_acc"][-1]}, f: {results["epoch_val_f"][-1]}, rationale scores: look, it\'s already a pain to duplicate this code. What do you want from me.')

            # if epoch_val_loss < best_val_loss:
            if epoch_val_total_loss < best_val_total_loss:
                logging.debug(f'Epoch {epoch} new best model with val loss {epoch_val_total_loss}')
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in mtl_token_identifier.state_dict().items()})
                best_epoch = epoch
                best_val_loss = epoch_val_total_loss
                torch.save(mtl_token_identifier.state_dict(), model_save_file)
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_loss': best_val_loss,
                    'done': 0
                }
                torch.save(epoch_data, epoch_save_file)
        if epoch - best_epoch > patience:
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    mtl_token_identifier.load_state_dict(best_model_state_dict)
    mtl_token_identifier = mtl_token_identifier.to(device=device)
    mtl_token_identifier.eval()

    def prepare_for_cl(input_data, keep_corrected_only=False):
        """
        Extract rationale prediction from document only.
        If keep_corrected_only=True keep only the annotation where the classification was correct
        :param input_data:
        :param keep_corrected_only:
        :return:
        """
        epoch_input_data = _prep_data_for_epoch(input_data, sampling_method)
        _, _, _, soft_pred_for_cl, hard_pred_for_cl, _, \
        pred_labels_for_cl, labels_for_cl = \
            make_mtl_token_preds_epoch(mtl_token_identifier, epoch_input_data, labels_mapping,
                                       token_mapping, batch_size, max_length, par_lambda,
                                       device, cls_criterion, exp_criterion, tensorize_model_inputs)
        hard_pred_for_cl = [h.cpu().tolist() for h in hard_pred_for_cl]
        hard_pred_for_cl = [h[len(d[1].query) + 2:] for h, d in zip(hard_pred_for_cl, epoch_input_data)]
        soft_pred_for_cl = [s[len(d[1].query) + 2:] for s, d in zip(soft_pred_for_cl, epoch_input_data)]
        train_ids = list(range(len(labels_for_cl)))
        if keep_corrected_only:
            labels_for_cl = [np.argmax(x) for x in labels_for_cl]
            pred_labels_for_cl = [np.argmax(x) for x in pred_labels_for_cl]
            train_ids = list(filter(lambda i: labels_for_cl[i] == pred_labels_for_cl[i],
                                    range(len(labels_for_cl))))
        return [(epoch_input_data[i], soft_pred_for_cl[i], hard_pred_for_cl[i]) for i in train_ids]

    train_machine_annotated = prepare_for_cl(evidence_train_data,False)
    eval_machine_annotated = prepare_for_cl(evidence_val_data, False)
    test_machine_annotated = prepare_for_cl(evidence_test_data, False)
    return mtl_token_identifier, results, \
           train_machine_annotated, eval_machine_annotated, test_machine_annotated
