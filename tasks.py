import config

from app import db
from celery import Celery, current_task, group, chord, chain
import app.models
import config
from functools import wraps

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
   
    patch = chord(group(patch_dataset.si(ds.id, ps.id)
                        for ps in ds.patchspecs), done.si())
        
    analyze = group(analyze_blob.si(blob.id,
                                    *[fs.id for fs in ds.featurespecs])
                    for blob in ds.blobs)
    chain(patch, analyze).apply_async()

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
    last_round = classifier.rounds[-1]
    round = app.models.Round(classifier = classifier,
                             number = last_round.number+1)
    db.session.add(round)

    for pq in last_round.queries:
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

