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
        fs.params = {"bins": 3}
        db.session.add(fs)

    fs = FeatureSpec.ifNew(name="TinyImage", cls="extract.TinyImage")
    if fs:
        db.session.add(fs)

    # fs = FeatureSpec.ifNew(name = 'CNN_CaffeNet_redux',
    #                                    cls = 'extract.CNN',
    #                                    params = {
    #                                    'use_reduce' : True,
    #                                    'ops' : ["subsample", "power_norm"],
    #                                    'output_dim' : 200,
    #                                    'alpha' : 2.5,
    #                                    'model' : 'caffenet',
    #                                    'def_fname' : "caffemodels/caffenet/train.prototxt",
    #                                    'weights_fname': "caffemodels/caffenet/weights.caffemodel",
    #                                    })
    # if fs:
    #     db.session.add(fs)

    # fs = FeatureSpec.ifNew(name = 'CNN_MobileNet_redux',
    #                                    cls = 'extract.CNN',
    #                                    params = {
    #                                    'use_reduce' : True,
    #                                    'ops' : ["subsample", "power_norm"],
    #                                    'output_dim' : 200,
    #                                    'alpha' : 2.5,
    #                                    'model' : 'mobilenet',
    #                                    'def_fname' : "caffemodels/mobilenet/mobilenet_v2_pydeploy.prototxt",
    #                                    'weights_fname': "caffemodels/mobilenet/mobilenet_v2.caffemodel",
    #                                    })
    # if fs:
    #     db.session.add(fs)

    # fs = FeatureSpec.ifNew(name = 'CNN_Places_redux',
    #                                    cls = 'extract.CNN',
    #                                    params = {
    #                                    'use_reduce' : True,
    #                                    'ops' : ["subsample", "power_norm"],
    #                                    'output_dim' : 200,
    #                                    'alpha' : 2.5,
    #                                    'model' : 'places',
    #                                    'def_fname' : "caffemodels/places/places205CNN_pydeploy_upgraded.prototxt",
    #                                    'weights_fname': "caffemodels/places/places205CNN_iter_300000_upgraded.caffemodel",
    #                                    'mean_fname': "caffemodels/places/places205CNN_mean.npy"
    #                                    })
    # if fs:
    #     db.session.add(fs)

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
