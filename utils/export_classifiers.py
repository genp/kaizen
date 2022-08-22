#!/usr/bin/env python
"""
This module creates a consistently labled dataset from the current
state of annotated instances and classifiers for ingestion by
an outside training pipeline.

Example syntax to run:
python utils/export_classifiers.py \
    --keywords-list keyword_1 keyword_1_bis \
    --keywords-list keyword_2_bis keyword_2 \
    --dataset-name "Ela's Airplanes" \
    --estimator-id 1
"""

import argparse
import datetime
import pickle
import random

import numpy as np

from app import db
from app.models import *


def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Training with Otto')
    parser.add_argument('--classifiers-list', type=int, nargs='+', action='append',
                        help="""list of list of classifiers where each
                        sublist refer to one single concept trained
                        multiple times. it overrides the keywords_list
                        argument""")
    parser.add_argument('--keywords-list', type=str, nargs='+', action='append',
                        help="""list of list of keywords where each
                        sublist refer to one single concept trained
                        multiple times""")
    parser.add_argument('--dataset-name', type=str, nargs='+',
                        help='name of the dataset we want to use to create the dict_dataset')
    parser.add_argument('--featurespec-ids', type=int, nargs='+',
                        help='featurespecs to use to represent the images')
    parser.add_argument('--estimator-id', type=str,
                        help='id of the estimator we want to use to instantiate the classifier')
    parser.add_argument('--attribute-names', type=str, nargs='+',
                        help='list of names of the attribute associated with each list of list')
    parser.add_argument('--only-examples', action='store_true',
                        help='flag to export one dict dataset per concept with only the examples')
    parser.add_argument('--subsample-ratio', type=float, default=-1,
                        help='positives = subsample-ratio * negatives')
    parser.add_argument('--hardneg', type=bool, default=False,
                        help='boolean flag for hard negative mining')
    parser.add_argument('--blob-root', type=str, default='',
                        help='root prefix for blob locations that should be removed')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='file prefix for output dict dataset file')    

    args = parser.parse_args()

    export_labeled_dataset(args)

def export_labeled_dataset(args):
    
    #Querying for the dataset we're going to work on
    datasets = db.session.query(Dataset).filter(Dataset.name.in_(args.dataset_name)).all()
    print(datasets)

    summary = {}
    clf_scores = {}
    
    for k, classifier_ids in enumerate(args.classifiers_list):
        # Decided to examine one fs at a time so a different
        # classifier and rounds get made for each different fs and
        # their predictions are not mixed
        final_round = None
        best_score = 0.0
        best_exp = ''
        best_fsid = ''
        for fs in args.featurespec_ids:
            fs = [fs]
            exporter = get_exporter(classifier_ids=classifier_ids, featurespec_ids=fs, dset=datasets[0])

            print(f'\n ****** Created new Classifier {exporter} for category {args.attribute_names[k]} ***** \n')
            
            fs_list = FeatureSpec.query.filter(FeatureSpec.id.in_(fs)).all()
            print(f'*** Using feature in fs_list: {fs_list} ***')

            exporter_round = {}
            exporter_round['all_ex'] = Round.query.filter_by(classifier=exporter, number=0).first()

            # Double check that all examples have the required features precalculated
            check_missing_example_feats(exporter, fs_list)

            # Calculate estimators using all examples
            print('*** Training estimators on all available examples ***')
            estimators, clf_scores['all_ex'] = exporter_round['all_ex'].trained_estimators(cur_round_only=True,
                                                                                           featurespecs=fs_list,
                                                                                           trainval_split=True,
                                                                                           output_scores=True)

            predict_round_export(exporter_round['all_ex'].id, fs, estimators, datasets)
            
            if args.subsample_ratio > 0:
                # Make a new round for the subsampled predictions
                print('*** Training estimators on subsampled examples ***')

                exporter_round['subsample'] = Round.query.filter(Round.classifier==exporter,
                                                    Round.number==100).first()

                if exporter_round['subsample'] is None:
                  exporter_round['subsample'] = Round(classifier = exporter, number=100)
                  db.session.add(exporter_round['subsample'])
                  db.session.commit()

                estimators, clf_scores['subsample'] = exporter_round['subsample'].subsample_estimator(args.subsample_ratio, cur_round_only=False,
                                                                                                      use_hard_negatives=False,
                                                                                                      featurespecs=fs_list, trainval_split=True)        
                
                predict_round_export(exporter_round['subsample'].id, fs, estimators, datasets)

              
                # get hard negative round
                if args.hardneg:
                    print('*** Training estimators on hard negative examples ***')
                    exporter_round['hard_negs'] = Round.query.filter(Round.classifier==exporter,
                                                                     Round.number==101).first()
                    if exporter_round['hard_negs'] is None:
                      exporter_round['hard_negs'] = Round(classifier = exporter, number=101)
                      db.session.add(exporter_round['hard_negs'])
                      db.session.commit()

                    estimators, clf_scores['hard_negs'] = exporter_round['hard_negs'].subsample_estimator(args.subsample_ratio, cur_round_only=False,
                                                                                                          use_hard_negatives=True,
                                                                                                          featurespecs=fs_list, trainval_split=True)        
                    
                    predict_round_export(exporter_round['hard_negs'].id, fs, estimators, datasets)

            # check which round has best val performance
            # Note: these val scores are not perfectly
            # comparable bc they are different val sets, this
            # is a hack
            for exp in clf_scores.keys():
                for fsid in clf_scores[exp].keys():
                    if clf_scores[exp][fsid]['val'] > best_score:
                        best_score = clf_scores[exp][fsid]['val']
                        final_round = exporter_round[exp]
                        best_exp = exp
                        best_fsid = fsid
                    
        print(f'*** Best classifier for this concept is {best_exp} with feature spec {best_fsid}, val score: {best_score:.5} ***')

        # Update labels for this category
        for pred in final_round.predictions:        
            # We add the predictions to the summary
            if pred.patch_id in summary.keys():
                summary[pred.patch_id][1][k] = pred.value > 0 
            else:
                summary[pred.patch_id] = (pred.patch.blob.location[len(args.blob_root):], np.zeros(max(len(args.attribute_names),2)), pred.patch.bbox)
                summary[pred.patch_id][1][k] = pred.value > 0

                # if this is only a one class dataset
                if len(args.attribute_names) == 1:
                    summary[pred.patch_id][1][k+1] = pred.value <= 0
          
    lbls = list(summary.values())
    val_idx = random.sample(range(len(lbls)), int(len(lbls)*.2))
    dict_dataset = {
        'attributes': args.attribute_names,
        'split':{
            'train': [lbls[i] for i in range(len(lbls)) if i not in val_idx] ,
            'val': [lbls[i] for  i in range(len(lbls)) if i in val_idx],
        },
    }    

    dt = str(datetime.date.today())

    savename = f'{args.save_prefix}_oscar_export_dict_labels_train_{dt}.pkl'
    print(f'Saving dataset file to {savename}.......')
    
    with open(savename, 'wb') as f:
        print(f.name)
        pickle.dump(dict_dataset, f)
    


def predict_round_export(r_id, featurespec_ids, estimators, datasets):
    """
    Mimics the tasks.round_predict function,
    making predictions for a specific round and precroping the patch to show on the active query web page
    """
    clf_round = Round.query.get(r_id)

    # We start by removing the existing predictions...
    for pred in clf_round.predictions:
        db.session.delete(pred)

    # ... and the patch queries
    for pq in clf_round.queries:
        db.session.delete(pq)

    for dataset in datasets:
        for pred in clf_round.predict(ds=dataset, val=False, featurespec_ids=featurespec_ids, estimators=estimators):
            db.session.add(pred)

    for pq in clf_round.choose_distributed_queries():
        db.session.add(pq)


    # TODO handle val predictions
    """    
    for vdset in exporter.dataset.val_dset:
        for vpred in round.predict(ds=vdset, val=True, featurespec_ids=featurespec_ids, estimators=estimators):
            db.session.add(vpred)
    """

    db.session.commit()
    tasks.precrop_round_results.run(clf_round.id)


def attach_examples_from_classifiers(clf_round, classifier_ids):
    '''
    Adds the examples from each classifier in classifier_ids to the clf_round.
    '''

    classifiers = list(Classifier.query.filter(Classifier.id.in_(classifier_ids)).all())
    print(f'*** Adding examples from these classifiers to new exporter classifier: {classifiers} ***')

    # TODO this is kind of slow. Could be replaced by a couple of faster raw SQL commands?
    for classifier in classifiers:
        for ex in classifier.examples:
            # if the current example has already been added to the
            # exporter, skip
            e = Example.ifNew(value = ex.value, patch = ex.patch, round = clf_round)
            if e:
                db.session.add(e)
                db.session.commit()

    return


def check_missing_example_feats(exporter, featurespecs):
    '''Generates features for the specified featurespecs if they are
    missing from the exporter's examples.  
    '''

    dset = exporter.dataset
    for fs in featurespecs:
        patches_w_feat = db.engine.execute(f'select patch_id from feature where spec_id = {fs.id}').fetchall()
        patches_w_feat = [x[0] for x in patches_w_feat]
        example_patches = db.engine.execute('select patch_id from example').fetchall()
        example_patches = [x[0] for x in example_patches]
        patches_wo_feat = list(set(example_patches) - set(patches_w_feat))
        if len(patches_wo_feat) > 0:
            dset.create_patch_features(patches_wo_feat, fs.id, batch_size=200)

    
def get_exporter(classifier_ids, featurespec_ids, dset=None):
    '''
    Creates or retrieves a Classifier containing all examples from the member classifier_ids

    classifier_ids: list of Classifiers ids (ints) that exist in the DB
    featurespec_ids: list of FeatureSpec ids to use to create this Classifier's Estimators
    dset: Dataset to attach this classifier to by default
    '''

    # Need a user to create a new classifier
    # Picking the first user from the DB
    u = User.query.filter(User.username == 'oscar').first()
    u.is_enabled = True;

    if not dset:
        # Need a dataset to create a new classifier
        # Picking the first user from the DB
        dset = Dataset.query.first()

    # Creating export classifier using an esitmator that may or maynot
    # be in the DB

    # If this type of estimator doesn't exist, make it 
    e = Estimator.query.filter(Estimator.cls == 'sklearn.svm.SVC').first()
    if not e:
        e = Estimator()
        e.cls = 'sklearn.svm.SVC'
        db.session.add(e)

    # If this Estimator exists, make sure it has the params we expect
    
    e.params = {
        'kernel':'linear', 
        'C':10, 
        'probability':True
    }
    db.session.commit()

    # Get the classifier for these classifier_ids and featurespec_ids
    # if it exists or else create it
    params = {'classifier_ids' : classifier_ids, 
              'featurespec_ids': featurespec_ids}
    exporter = Classifier.query.filter(Classifier.dataset == dset,
                                       Classifier.keyword == None,
                                       Classifier.estimator == e,
                                       Classifier.owner == u,
                                       Classifier.params == params).first()
    if not exporter:
        exporter = Classifier(dataset = dset,
                           keyword = None,
                           estimator = e,
                          owner=u,
                         params = {'classifier_ids' : classifier_ids, 
                                   'featurespec_ids': featurespec_ids})
        db.session.add(exporter)
        db.session.commit()

        # Since this exporter doesn't already exist, add all the
        # examples from all member classifiers
        attach_examples_from_classifiers(exporter.latest_round,
                                         exporter.params['classifier_ids'])
        
    return exporter
    
    

if __name__ == '__main__':
    main()
