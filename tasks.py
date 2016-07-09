#!/usr/bin/env python
import sys
import csv

from celery import Celery, current_task, group, chord, chain
from functools import wraps
import celery.registry as registry

from app import db
import app.models
import config


celery = Celery('tasks',
                broker="sqla+"+config.SQLALCHEMY_DATABASE_URI,
                backend="db+"+config.SQLALCHEMY_DATABASE_URI)

# Get rid of pickle (insecure, causes long deprecation warning at startup)
celery.conf.CELERY_TASK_SERIALIZER = 'json'
celery.conf.CELERY_ACCEPT_CONTENT = ['json', 'msgpack', 'yaml']


# Need to figure out some details here.  Currently, this file uses the
# SQLAlchemy object from Flask to access db.  That's probably wrong.
# See:
# http://prschmid.blogspot.com/2013/04/using-sqlalchemy-with-celery-tasks.html

class SqlAlchemyTask(celery.Task):
    """An abstract Celery Task that ensures that the connection the the
    database is closed on task completion"""
    abstract = True
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        db.session.remove()

# Decorator to turn a task into a task that tries to retry
# itself. It's a bit ugly, but we often want to run some asynchronous
# task on an on an object that we've only just created (but have not
# committed).  By setting up tasks to retry, we'll eventually get the
# work done, generally on the first retry, since it will run after the
# commit.
def that_retries(task):
    @wraps(task)
    def retrying(*args, **kwargs):
        try:
            task(*args, **kwargs)
        except Exception as e:
            current_task.retry(exc = e, countdown=30)
    return retrying


# Celery won't let you chain groups. If you do, the first group
# becomes a chord, trying to feed its result into the second group
# - which doesn't work.
    
# So, we make the first group into a chord that feeds a dummy task.
# The chord can be can used as the first member of a chain.

@celery.task
def done(*args, **kwargs):
    '''A no-op task, used to work around the chord/group/chain issue'''
    return "DONE"


def if_dataset(ds):
    if ds:
        dataset.delay(ds.id)

@celery.task
@that_retries
def dataset(ds_id):
    ds = app.models.Dataset.query.get(ds_id)
    ks = app.models.Keyword.query.filter(Keyword.dataset_id==ds.id).all()

    patch = chord(group(patch_dataset.si(ds.id, ps.id)
                        for ps in ds.patchspecs), done.si())
        
    analyze = group(analyze_blob.si(blob.id,
                                    *[fs.id for fs in ds.featurespecs])
                    for blob in ds.blobs)
    examples = chord(group(add_examples.si(k.id)
                           for k in ks), done.si())
    chain(patch, analyze, examples).apply_async()

@celery.task
@that_retries
def patch_dataset(ds_id, ps_id):
    ds = app.models.Dataset.query.get(ds_id)
    ps = app.models.PatchSpec.query.get(ps_id)
    for patch in ps.create_dataset_patches(ds):
        db.session.add(patch)
    db.session.commit()

@celery.task
@that_retries
def analyze_blob(blob_id, *fs_ids):
    blob = app.models.Blob.query.get(blob_id)
    # TODO: Use an in query so that this is one query, not one per fs
    for fs in [app.models.FeatureSpec.query.get(fs_id) for fs_id in fs_ids]:
        for feat in fs.create_blob_features(blob):
            db.session.add(feat)
    db.session.commit()

@celery.task
@that_retries
def add_examples(k_id):
    k = app.models.Keyword.query.get(k_id)
    # read definition file
    for row in csv.reader(k.defn_file):
        # create examples for each row        
        blob_name, x, y, h, w, val = row # TODO format this for the expected types

        # check if blob exists
        blob = app.models.Blob.query.filter(app.models.Blob.location.like('%{}'.format(blob_name)))
        if blob is None:
            print 'Cannot add example for file {}'.format(blob_name)
            # TODO: add log entry
        # check if patch exists
        patch = app.models.Patch.query.\
                filter(app.models.Patch.blob==blob).\
                filter(app.models.Patch.x==x).\
                filter(app.models.Patch.y==y).\
                filter(app.models.Patch.h==h).\
                filter(app.models.Patch.w==w).first()

        # create new patch and feature
        if patch is None:
            patch = app.models.Patch(blob=blob,
                          x=x,
                          y=y,
                          width=w,
                          height=h,
                          fliplr=False, rotation=0.0)
            db.session.add()
            db.session.commit()

            ds = k.dataset
            fs_ids = [fs.id for fs in ds.featurespecs]
            for fs in [app.models.FeatureSpec.query.get(fs_id) for fs_id in fs_ids]:
                for feat in fs.create_patch_features(patch):
                    db.session.add(feat)
            db.session.commit()

        # add example to db
        ex = app.models.Example(value=val,patch=patch,keyword=k)
        db.session.add(ex)
        db.session.commit()



def if_classifier(c):
    if c:
        classifier.delay(c.id)

@celery.task
@that_retries
def classifier(c_id):
    c = app.models.Classifier.query.get(c_id)
    kw = c.keyword
    ds = c.dataset

    # Start the classifier with seeds from the keyword
    negative = False;
    zero = c.rounds[0]
    for ex in kw.seeds:
        e = app.models.Example(value = ex.value, patch = ex.patch, round = zero)
        db.session.add(e)

        # We added at least one negative value from the seeds
        if not ex.value:
            negative = True

        # Calculate features for the example patches (as needed)
        for fs in ds.featurespecs:
            feat = fs.create_patch_feature(ex.patch)
            if feat:
                db.session.add(feat)

    # If no negative seeds, cross fingers and add one "random" patch
    # to serve as negative. It will already have the features
    # calculated, since it comes from the dataset.

    # It would be preferable to only do this when the Estimator in use
    # really needs negative examples to work well (or add interface to
    # accept negatives, and require them in such cases).
    
    if not negative:
        patch = ds.blobs[0].patches[0]
        e = app.models.Example(value = False, patch = patch, round = zero)
        db.session.add(e)

    predict_round(zero.id)
    db.session.commit()

@celery.task
@that_retries
def advance_classifier(c_id):
    classifier = app.models.Classifier.query.get(c_id)
    latest_round = classifier.latest_round
    round = app.models.Round(classifier = classifier,
                             number = latest_round.number+1)
    db.session.add(round)

    for pq in latest_round.queries:
        value = pq.responses[0].value # should be a vote, avg, etc
        ex = app.models.Example(value=value, patch=pq.patch, round=round)
        db.session.add(ex)

    predict_round(round.id)
    db.session.commit();


@celery.task
@that_retries
def predict_round(r_id):
    round = app.models.Round.query.get(r_id)

    for pred in round.predict():
        db.session.add(pred)

    for pq in round.choose_queries():
        db.session.add(pq)
        
    db.session.commit()

@celery.task
@that_retries
def detect(d_id):
    detect = app.models.Detection.query.get(d_id)
    dense = app.models.PatchSpec.query.filter_by(name='Dense').one()
    # Patch the blob
    for patch in dense.create_blob_patches(detect.blob):
        db.session.add(patch)
    for c in app.models.Classifier.query.all():
        print c
        # Create features for the patches
        for fs in c.dataset.featurespecs:
            print " ",fs
            for f in fs.create_blob_features(detect.blob):
                print "  ",f
                db.session.add(f)
        # Test the patches of the blob, saving Predictions
        for p in c.latest_round.detect(detect.blob):
            print " p"
            db.session.add(p)
    db.session.commit()

if __name__ == "__main__":
    function = sys.argv[1]
    ids = [int(s) for s in sys.argv[2:]]
    print function, ids
    task = registry.tasks["tasks."+function]
    task(*ids)
    
