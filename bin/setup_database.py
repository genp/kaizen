#!/usr/bin/env python

import json

from app import db, APPNAME

# Preload the DB so we can drop it any time.
from app.models import User, PatchSpec, FeatureSpec, Estimator

# Imports for db setup
from migrate.versioning import api
from config import SQLALCHEMY_DATABASE_URI
from config import SQLALCHEMY_MIGRATE_REPO
import os.path


def create_db():
    db.create_all()
    if not os.path.exists(SQLALCHEMY_MIGRATE_REPO):
        api.create(SQLALCHEMY_MIGRATE_REPO, 'database repository')
        api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    else:
        api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO,
                            api.version(SQLALCHEMY_MIGRATE_REPO))


def add_default_user():
    u = User.ifNew(username=APPNAME)
    if u:
        u.password = APPNAME
        u.is_enabled = True
        db.session.add(u)

    db.session.commit()


def add_default_patch_specs():

    ps = PatchSpec.ifNew(name="Default")
    if ps:
        ps.width = -1
        ps.height = -1
        ps.xoverlap = 0.0
        ps.yoverlap = 0.0
        ps.scale = 100.0
        ps.fliplr = False
        db.session.add(ps)

    ps = PatchSpec.ifNew(name="Sparse")
    if ps:
        ps.width = 400
        ps.height = 400
        ps.xoverlap = 0.1
        ps.yoverlap = 0.1
        ps.scale = 3.0
        db.session.add(ps)

    ps = PatchSpec.ifNew(name="Dense")
    if ps:
        ps.width = 200
        ps.height = 200
        ps.xoverlap = 0.5
        ps.yoverlap = 0.5
        ps.scale = 2
        ps.fliplr = True
        db.session.add(ps)

    ps = PatchSpec.ifNew(name="LessDense_NoFlip")
    if ps:
        ps.width = 200
        ps.height = 200
        ps.xoverlap = 0.25
        ps.yoverlap = 0.25
        ps.scale = 2
        ps.fliplr = True
        db.session.add(ps)

    db.session.commit()


def add_default_feature_specs():
    fs = FeatureSpec.ifNew(name="RGB", cls="extract.ColorHist")
    if fs:
        db.session.add(fs)

    fs = FeatureSpec.ifNew(name="RGB coarse", cls="extract.ColorHist")
    if fs:
        # Note: sqlalchemy can't seem to add the json type columns in the constructor,
        # so adding them after the FS object is created.
        fs.params = {"bins": 3}
        db.session.add(fs)

    fs = FeatureSpec.ifNew(name="TinyImage", cls="extract.TinyImage")
    if fs:
        db.session.add(fs)

    fs = FeatureSpec.ifNew(name = 'timm_vit',
                           cls = 'extract.TimmModel')
    if fs:
        fs.params = { 'use_reduce' : False,
                      'model' : "vit_base_patch16_224",
                     }
        db.session.add(fs)

    fs = FeatureSpec.ifNew(name = 'timm_wide_resnet_redux',
                           cls = 'extract.TimmModel')
    if fs:
        fs.params = {'use_reduce' : True,
                     'ops' : ["subsample", "power_norm"],
                     'output_dim' : 100,
                     'alpha' : 2.5,
                     'model' : "wide_resnet50_2",
                     }
        db.session.add(fs)

    db.session.commit()


def add_default_estimators():
    # e = Estimator.ifNew(cls = 'turicreate squeezenet')
    # if e:
    #     e.params = {'model_name': 'squeezenet_v1.1'}
    #     db.session.add(e)

    # e = Estimator.ifNew(cls = 'turicreate resnet-50')
    # if e:
    #     e.params = {'model_name': 'resnet-50'}
    #     db.session.add(e)

    # e = Estimator.ifNew(cls = 'sklearn.neighbors.KNeighborsRegressor')
    # if e:
    #     e.params = {'weights' : 'distance', 'n_neighbors': 2}
    #     db.session.add(e)

    e = Estimator.ifNew(cls="sklearn.svm.LinearSVC")
    if e:
        e.params = {}
        db.session.add(e)

    db.session.commit()


def setup_database_defaults():
    create_db()
    add_default_user()
    add_default_patch_specs()
    add_default_feature_specs()
    add_default_estimators()

if __name__ == "__main__":
    from config import APPNAME

    print(f"Setting up database defaults for {APPNAME}")

    setup_database_defaults()
