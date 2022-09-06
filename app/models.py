#!/usr/bin/env python
import importlib
import multiprocessing
import os
import random
import resource
import time
import urllib
import collections

from sqlalchemy.sql.expression import func
from flask import url_for
from flask_login import UserMixin
from more_itertools import chunked
import pandas as pd
import imageio
from scipy.ndimage.interpolation import rotate
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import skvideo.io
from sqlalchemy import orm
import sqlalchemy.types as types
import boto3
import numpy as np
import psutil
import cv2

from app import app, db
import config
import extract
import tasks

s3 = boto3.resource("s3")


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, nullable=False, unique=True)
    # if password is null, then can't login in, except by mturk-like bypass
    password = db.Column(db.String, nullable=True)
    reset_password_token = db.Column(db.String, nullable=True)

    email = db.Column(db.String, nullable=True, unique=True)
    confirmed_at = db.Column(db.DateTime())

    is_enabled = db.Column(db.Boolean, nullable=False, default=False)

    def is_active(self):
        return self.is_enabled

    @staticmethod
    def find(username):
        return User.query.filter_by(username=username).first()

    @classmethod
    def ifNew(model, **kwargs):
        if not model.query.filter_by(**kwargs).first():
            return model(**kwargs)

    def check_password(self, given):
        return given == self.password

    def __repr__(self):
        return model_debug(self)


dataset_x_blob = db.Table(
    "dataset_x_blob",
    db.Model.metadata,
    db.Column("dataset_id", db.Integer, db.ForeignKey("dataset.id")),
    db.Column("blob_id", db.Integer, db.ForeignKey("blob.id")),
)


def s3_url(location):
    assert location[:5] == "s3://"
    (bucket, key) = location[5:].split("/")
    return "http://%s.s3.amazonaws.com/%s" % (bucket, key)


def static_url(location):
    prefix = config.approot + "/app/static/"
    assert location.startswith(prefix)
    return url_for("static", filename=location[len(prefix) :])


def clean_cache(s):
    dir = config.CACHE_DIR
    now = time.time()
    for fn in os.listdir(dir):
        complete = os.path.join(dir, fn)
        last_access = os.path.getatime(complete)
        if now - last_access > 24 * 60 * 60:
            os.remove(complete)


class Blob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ext = db.Column(db.String)
    mime = db.Column(db.String)
    location = db.Column(db.String)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    frame_rate = db.Column(db.Float, nullable=False, default=0.0)

    URL_MAP = {
        config.approot + "/app/static/": static_url,
        "s3://": s3_url,
    }

    def __init__(self, location):
        self.location = location
        self.ext = os.path.splitext(location)[1]

        self.latitude, self.longitude = (None, None)
        if self.ext == ".jpg":
            self.mime = "image/jpeg"
        elif self.ext == ".mp4":
            self.mime = "video/mp4"
            video_cv2 = cv2.VideoCapture(self.materialize())
            frame_rate = video_cv2.get(cv2.CAP_PROP_FPS)
            if frame_rate > 0.0:
                self.frame_rate = frame_rate
        else:
            self.mime = "image/" + self.ext.replace(".", "")

        self.img = None
        self.vid = None

    @orm.reconstructor
    def init_on_load(self):
        self.img = None
        self.vid = None

    def open(self):
        return open(self.materialize(), "rb")

    def materialize(self):
        return self.location

        assert self.id
        complete = cache_fname(config.CACHE_DIR, self.filename)

        if not os.path.exists(complete):
            urllib.urlretrieve(self.url, complete)
        return complete

    def read_lat_lon(self):
        return (0, 0)  # TODO BROKEN exif.get_lat_lon(exif.get_data(self.materialize()))

    @property
    def features(self):
        for patch in self.patches:
            for feature in patch:
                yield feature

    @property
    def image(self):
        # if self.img is not None:
        #  return self.img
        try:
            img = imageio.imread(self.location)
            return img
        except IOError as e:
            print("Could not open image file for {}".format(self))
            return None

    @property
    def is_video(self):
        return self.frame_rate > 0.0

    @property
    def video(self):
        try:
            return skvideo.io.vread(self.materialize())
        except IOError as e:
            print("Could not open video file for {}".format(self))
            return None

    def reset(self):
        self.img = None
        self.vid = None

    @property
    def local(self):
        return self.location[0] == "/" and os.path.exists(self.location)

    @property
    def on_s3(self):
        return self.location.startswith("s3://")

    @property
    def url(self):
        url = self.location
        for prefix, change in list(Blob.URL_MAP.items()):
            if url.startswith(prefix):
                return change(url)
        return url

    @property
    def filename(self):
        return "blob-" + str(self.id) + self.ext

    BUCKET = config.APPNAME + "-blobs"

    def migrate_to_s3(self):
        if self.on_s3:
            return
        with self.open() as body:
            s3.Bucket(Blob.BUCKET).put_object(Key=self.filename, Body=body)
        self.location = "https://s3.amazonaws.com/%s/%s" % (Blob.BUCKET, self.filename)
        db.session.commit()

    def __repr__(self):
        return "Blob#" + str(self.id) + ":" + self.location


dataset_x_patchspec = db.Table(
    "dataset_x_patchspec",
    db.Model.metadata,
    db.Column("dataset_id", db.Integer, db.ForeignKey("dataset.id")),
    db.Column("patchspec_id", db.Integer, db.ForeignKey("patch_spec.id")),
)


class PatchSpec(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    # Starting size (smallest patches)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)

    # Fraction of patch to keep while sliding over for next patch
    xoverlap = db.Column(db.Float, nullable=False)  # 0 < xoverlap <= 1
    # Fraction of patch to keep while sliding down for next row of patches
    yoverlap = db.Column(db.Float, nullable=False)  # 0 < yoverlap <= 1

    # Scale up patches until too big, by this factor
    scale = db.Column(db.Float, nullable=False)

    # Make mirrored patches too
    fliplr = db.Column(db.Boolean, nullable=False, default=False)

    @classmethod
    def ifNew(model, **kwargs):
        if not model.query.filter_by(**kwargs).first():
            return model(**kwargs)

    @property
    def url(self):
        return "/patchspec/" + str(self.id)

    def __repr__(self):
        upto = " by " + str(self.scale)
        if self.xoverlap == self.yoverlap:
            overlap = str(self.xoverlap) + " overlap"
        else:
            overlap = str(self.yoverlap) + "/" + str(self.xoverlap) + " overlap"
        return (
            self.name
            + ": "
            + str(self.height)
            + "x"
            + str(self.width)
            + upto
            + " with "
            + overlap
        )

    def create_blob_patches(self, blob):
        print("Creating patches for {}".format(blob))
        print(
            "Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

        data = None
        if blob.is_video:
            data = blob.video
            assert len(data.shape) == 4, data.shape
        else:
            data = blob.image
            assert len(data.shape) >= 2, data.shape

        # dims of image: (H x W x 3 (RGB))
        # dims of video: (n_frames x H x W x 3 (RGB))
        image_height = data.shape[1] if blob.is_video else data.shape[0]
        image_width = data.shape[2] if blob.is_video else data.shape[1]
        image_shape = (image_height, image_width)

        try:
            # if this PatchSpec has w, h = (-1, -1) it is meant to crop the whole blob
            w = self.width if self.width > 0 else image_width
            h = self.height if self.height > 0 else image_height
            pind = 0
            while w < image_width and h < image_height:
                for patch in self.create_sized_blob_patches(blob, image_shape, w, h):
                    pind += 1
                    yield patch
                w = int(w * self.scale)
                h = int(h * self.scale)

            # if patch size is larger than blob, create one patch that is the whole image
            if w >= image_width and h >= image_height:
                for patch in self.create_sized_blob_patches(
                    blob, image_shape, image_width, image_height
                ):
                    pind += 1
                    yield patch

        except IndexError:
            print("Index Error for {}".format(blob))
            print("Error on Patch #{}".format(pind))
            print("Blob's Image Shape: {}".format(image_shape))
            if blob.is_video:
                print("Blob's Video Shape: {}".format(blob.video.shape))
            print(
                "Memory usage: %s (kb)"
                % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            blob.reset()
            print("*** Reseting blob image ***")
            print(
                "Memory usage: %s (kb)"
                % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            return
        except AttributeError:
            print("Could not load image from %s" % blob.location)
            blob.reset()
            print("*** Reseting blob image ***")
            print(
                "Memory usage: %s (kb)"
                % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            return

    def create_sized_blob_patches(self, blob, size, width, height):

        # Patch of the whole blob
        if width == size[1] and height == size[0]:
            xstep, ystep, xdelta, ydelta, left_indent, top_indent = (1, 1, 1, 1, 0, 0)

        else:
            # How far each slide will go
            xstep = int(width * (1 - self.xoverlap))
            ystep = int(height * (1 - self.yoverlap))

            # The available room for sliding
            xdelta = size[1] - width
            ydelta = size[0] - height

            # How many slides, and how much unused "slide space"
            xsteps, extra_width = divmod(xdelta, xstep)
            ysteps, extra_height = divmod(ydelta, ystep)

            # Divy up the unused slide space, to lose evenly at edges
            left_indent = extra_width // 2
            top_indent = extra_height // 2

        for x in range(left_indent, xdelta, xstep):
            for y in range(top_indent, ydelta, ystep):
                yield Patch.ensure(
                    blob=blob,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    fliplr=False,
                    frame_rate=blob.frame_rate,
                )
                if not self.fliplr:
                    continue
                yield Patch.ensure(
                    blob=blob,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    fliplr=True,
                    frame_rate=blob.frame_rate,
                )


dataset_x_featurespec = db.Table(
    "dataset_x_featurespec",
    db.Model.metadata,
    db.Column("dataset_id", db.Integer, db.ForeignKey("dataset.id")),
    db.Column("featurespec_id", db.Integer, db.ForeignKey("feature_spec.id")),
)


class FeatureSpec(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    cls = db.Column(db.String, nullable=False)
    params = db.Column(types.JSON)

    def __init__(self, **kwargs):
        super(FeatureSpec, self).__init__(**kwargs)
        self.instantiate()

    @classmethod
    def ifNew(model, **kwargs):
        if not model.query.filter_by(**kwargs).first():
            return model(**kwargs)

    @orm.reconstructor
    def instantiate(self):
        parts = self.cls.split(".")
        module = ".".join(parts[:-1])
        classname = parts[-1]
        self.instance = eval("importlib.import_module('%s').%s()" % (module, classname))
        if not self.params:
            self.params = {}
        self.instance.set_params(**self.params)

    @property
    def simple_class(self):
        return self.cls.split(".")[-1]

    def __repr__(self):
        return self.simple_class + "(" + str(self.params) + ")"

    @property
    def url(self):
        return "/featurespec/" + str(self.id)

    def analyze_blob(self, blob, batch_size=256):
        iter = 0
        for patches in chunked(self.undone_patches(blob), batch_size):
            print("calculating {}x{} patch features for {}".format(iter, blob, batch_size))
            iter += 1
            imgs = [p.image for p in patches]
            feats = self.instance.extract_many(imgs)
            if len(patches) == 1:
                feats = [feats]
            assert len(patches) == len(
                feats
            ), f"The number of patches {len(patches)} and features {len(feats)} are different for {blob}"
            for idx, f in enumerate(feats):
                yield Feature(patch=patches[idx], spec=self, vector=f)

    def undone_patches(self, blob):
        for p in blob.patches:
            if Feature.query.filter_by(patch=p, spec=self).count() == 0:
                yield p

    def analyze_patch(self, patch):
        if Feature.query.filter_by(patch=patch, spec=self).count() > 0:
            return None
        return Feature(
            patch=patch, spec=self, vector=self.instance.extract(patch.image)
        )


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    owner = db.relationship("User", backref=db.backref("datasets", lazy="dynamic"))

    blobs = db.relationship("Blob", secondary=dataset_x_blob)
    patchspecs = db.relationship(
        "PatchSpec", secondary=dataset_x_patchspec, backref="datasets"
    )
    featurespecs = db.relationship(
        "FeatureSpec", secondary=dataset_x_featurespec, backref="datasets"
    )

    is_train = db.Column(db.Boolean, nullable=False, default=True)

    def __init__(self, **kwargs):
        super(Dataset, self).__init__(**kwargs)
        # if self.owner is None and current_user.is_authenticated:
        #   self.owner = current_user

    def create_blob_patches(self, blob):
        """
        Extracts patches from blob for all PatchSpecs attached to this Dataset.
        """
        for ps in self.patchspecs:
            print(ps)
            for p in ps.create_blob_patches(blob):
                if p:
                    db.session.add(p)
                    db.session.commit()
                    print(p)
        print("All patches extracted!")

    def create_blob_features(self, blob, batch_size=256):
        print("calculating features for {}".format(blob))

        # Make sure all patches have been extracted
        self.create_blob_patches(blob)

        for fs in self.featurespecs:
            print(fs)
            feats = fs.analyze_blob(blob)
            for feat in chunked(feats, batch_size):
                print(
                    "Memory usage: %s (kb)"
                    % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )
                for f in feat:
                    db.session.add(f)
                db.session.commit()
                if fs.instance.__class__ is extract.CNN:
                    fs.instance.del_networks()
                print(
                    "Memory usage: %s (kb)"
                    % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )

    def create_patch_features(self, p_ids, fs_id, batch_size=256):
        """
        helper function for batch extracting features for a batch of patches
        """
        fs = FeatureSpec.query.get(fs_id)
        feats = fs.instance.extract_many([Patch.query.get(pid).image for pid in p_ids])
        if len(p_ids) == 1:
            feats = [feats]
        assert len(p_ids) == len(
            feats
        ), f"The number of patches {len(patches)} and features {len(feats)} are different for featurespec {fs.name} in dataset {self.name}"
        for idx, f in enumerate(feats):
            db.session.add(
                Feature(patch=Patch.query.get(p_ids[idx]), spec=fs, vector=f)
            )
        db.session.commit()
        if fs.instance.__class__ is extract.CNN:
            fs.instance.del_networks()
        return p_ids

    def create_all_patch_features(self, batch_size=256, fs_ids=False):
        """
        For all patches currently associated with this dataset,
        calculate all attach featurespecs' features.
        """
        try:
            nump = len(config.GPU_DEVICE_IDS)
        except:
            nump = 1

        if fs_ids is False:
            fs_ids = [fs.id for fs in self.featurespecs]
        for fs_id in fs_ids:
            # get all patches in this dataset that don't have this feature
            cmd = (
                "select id from patch "
                + "where blob_id in "
                + f"(select blob_id from dataset_x_blob where dataset_id = {self.id}) "
                + f"and id not in (select patch_id from feature where spec_id = {fs_id})"
            )
            print(cmd)
            all_patch_ids = db.engine.execute(cmd).fetchall()
            all_patch_ids = [ap[0] for ap in all_patch_ids]
            print(
                f"Need to calculate fs id {fs_id} for {len(all_patch_ids)} patches..."
            )
            for p_ids in chunked(all_patch_ids, batch_size):
                self.create_patch_features(p_ids, fs_id, batch_size)

            print(
                f"All features for FeatureSpec {fs_id} calculated for all patches in dataset {self.name}"
            )
        return []

    def unanalyzed_blob_ids(self):
        un_blobs = [row[0] for row in
                     db.engine.execute('select id from blob where id not in ' +
                          '(select blob_id from patch) and id in ' +
                          f'(select blob_id from dataset_x_blob where dataset_id = {self.id});').all()]
        return un_blobs

    #TODO: consider making patch_x_dataset table and feature_x_dataset table
    def patch_ids(self,limit=None):
        if limit:
            return db.engine.execute(f'select id from patch where blob_id in (select blob_id from dataset_x_blob where dataset_id = {self.id}) limit {limit}').all()
        else:
            return db.engine.execute(f'select id from patch where blob_id in (select blob_id from dataset_x_blob where dataset_id = {self.id})').all()

    def patches(self, limit=None):
        for p_id in self.patch_ids(limit):
            yield Patch.query.get(p_id)

    def feature_ids(self, limit=None):
        if limit:
            # TODO: this may need to be changed to include order by RANDOM ()
            return db.engine.execute('select id from feature where patch_id in ' +
                                     f'(select id from patch where blob_id in (select blob_id from dataset_x_blob where dataset_id = {self.id})) limit {limit}').all()
        else:
            return db.engine.execute('select id from feature where patch_id in ' +
                                     f'(select id from patch where blob_id in (select blob_id from dataset_x_blob where dataset_id = {self.id}))').all()

    def features(self, limit=None):
        for f_id in self.feature_ids(limit):
            yield Feature.query.get(f_id)

    def migrate_to_s3(self):
        for blob in self.blobs:
            blob.migrate_to_s3()

    @property
    def url(self):
        return url_for("dataset", id=self.id)

    def __repr__(self):
        return model_debug(self)

    @property
    def images(self):
        return len(self.blobs)


# Monkey-patch to make a foreign key reference to a dataset from another dataset
# Use case: a val or test dataset needs to be associate with the corresponding train dataset
Dataset.train_dset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"))
Dataset.train_dset = db.relationship(
    "Dataset", backref=db.backref("val_dset", lazy="dynamic"), remote_side=Dataset.id
)


class Keyword(db.Model):
    """
    Keywords can be created individually or be asscociated with a dataset
    When associated with a dataset, the keyword will have a defn_file that contains
    the location of a csv listing the images, patch locations, and label values of the
    examples associated with this keyword.
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    owner = db.relationship("User", backref=db.backref("keywords", lazy="dynamic"))

    geoquery_id = db.Column(db.Integer, db.ForeignKey("geo_query.id"))
    geoquery = db.relationship(
        "GeoQuery", backref=db.backref("keywords", lazy="dynamic")
    )

    defn_file = db.Column(db.String)

    def __init__(self, **kwargs):
        super(Keyword, self).__init__(**kwargs)
        # if self.owner is None and current_user.is_authenticated:
        #   self.owner = current_user

    @property
    def url(self):
        return url_for("keyword", id=self.id)

    def __repr__(self):
        return model_debug(self)


class GeoQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    @property
    def url(self):
        return "/geo/" + self.name

    def __repr__(self):
        return model_debug(self)


estimators = [
    "sklearn.neighbors.KNeighborsRegressor",
    "sklearn.neighbors.RadiusNeighborsRegressor",
    "sklearn.linear_model.LinearRegression",
]


class Estimator(db.Model):
    """An estimator is an object that describes a type of ML model,
    with the associated parameters for calling traiing, test, prediction."""

    id = db.Column(db.Integer, primary_key=True)
    cls = db.Column(db.String, nullable=False)
    params = db.Column(types.JSON)

    @classmethod
    def ifNew(model, **kwargs):
        if not model.query.filter_by(**kwargs).first():
            return model(**kwargs)

    def instantiate(self):
        parts = self.cls.split(".")
        module = ".".join(parts[:-1])
        classname = parts[-1]
        instance = eval("importlib.import_module('%s').%s()" % (module, classname))
        if self.params:
            instance.set_params(**self.params)
        return instance

    @property
    def simple_class(self):
        return self.cls.split(".")[-1]

    def __repr__(self):
        return self.simple_class + "(" + str(self.params) + ")"


class Classifier(db.Model):
    """A Classifier associates a dataset with a particular keyword search.
    It has been trained by examples, in Rounds.  The first round is
    prepared from the examples provided by the Keyword.  The dataset is
    then ranked based on those examples, and a set of PatchQueries are
    proposed for human classification.  Their answers create the
    examples for the next Round.

    Training the classifier from examples and then ranking the dataset
    to choose PatchQueries are compute intensive operations performed by
    background jobs, so a Classifier's latest Round may exist in an
    incomplete state until those jobs are complete.
    """

    id = db.Column(db.Integer, primary_key=True)

    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    owner = db.relationship("User", backref=db.backref("classifiers", lazy="dynamic"))

    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"), nullable=False)
    dataset = db.relationship(
        "Dataset", backref=db.backref("classifiers", lazy="dynamic")
    )

    keyword_id = db.Column(db.Integer, db.ForeignKey("keyword.id"), nullable=True)
    keyword = db.relationship(
        "Keyword", backref=db.backref("classifiers", lazy="dynamic")
    )

    estimator_id = db.Column(db.Integer, db.ForeignKey("estimator.id"), nullable=False)
    estimator = db.relationship("Estimator")

    is_ready = db.Column(db.Boolean, nullable=False, default=False)
    params = db.Column(types.JSON)

    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        # if self.owner is None and current_user.is_authenticated:
        #   self.owner = current_user
        zero = Round(classifier=self)

    @property
    def examples(self):
        for round in self.rounds:
            for example in round.examples:
                yield example

    def examples_upto(self, r):
        for round in self.rounds:
            for example in round.examples:
                yield example
            if r == round:
                return

    @property
    def latest_round(self):
        return self.rounds[-1]

    @property
    def url(self):
        return url_for("classifier", id=self.id)

    def __repr__(self):
        return model_debug(self)

    def attach_examples(self):
        if self.params["classifier_ids"]:
            classifiers = list(
                Classifier.query.filter(
                    Classifier.id.in_(self.params["classifier_ids"])
                )
            )
            print([x.id for x in classifiers])
            for classifier in classifiers:
                for ex in classifier.examples:
                    if (
                        Example.query.filter_by(
                            value=ex.value, patch=ex.patch, round=self.latest_round
                        ).first()
                        is not None
                    ):
                        pass
                    else:
                        e = Example.ifNew(
                            value=ex.value, patch=ex.patch, round=self.latest_round
                        )
                        db.session.add(e)
            db.session.commit()


class Round(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer, nullable=False, default=0)

    classifier_id = db.Column(
        db.Integer, db.ForeignKey("classifier.id"), nullable=False
    )
    classifier = db.relationship(
        "Classifier",
        backref=db.backref("rounds", order_by="Round.number", lazy="dynamic"),
    )

    location = db.Column(db.String)

    # to keep first_incorrect and last_correct of each round
    notes = db.Column(db.String)

    def predict(self, ds=None, val=False, featurespec_ids=None, estimators=None, limited_number_of_features_to_evaluate=None):

        """Yield Predictions for all Patches in the Dataset, as of this Round.
        Uses estimators trained on examples for each FeatureSpec to make
        predictions against its Features.
        """
        classifier = self.classifier
        if estimators is None:
            estimators = self.trained_estimators()

        if ds is None:
            ds = classifier.dataset

        #TODO: there might be a way to make this loop run faster if features are evaluated in batches instead of one at a time
        for feature in ds.features(limit=limited_number_of_features_to_evaluate):
            if featurespec_ids is None or feature.spec_id in featurespec_ids:
                # Note: this only works for sklearn estimators...
                try:
                    value = estimators[feature.spec.id].decision_function(
                        feature.nparray
                    )
                    if not val:
                        yield Prediction(
                            value=value,
                            feature=feature,
                            patch_id=feature.patch_id,
                            round=self,
                        )
                    else:
                        yield ValPrediction(
                            value=value,
                            feature=feature,
                            patch_id=feature.patch_id,
                            round=self,
                        )
                except IndexError as e:
                    print(f"IndexError!")
                    print(f"estimnator keys: {estimators.keys()}")
                    print(
                        f"estimators[feature.spec.id].decision_function(feature.nparray) {feature.spec.id} {feature.nparray.shape}"
                    )
                    pass
                except KeyError as e:
                    print(
                        f"KeyError! There is no estimator for this FeatureSpec {feature.spec.id}"
                    )
                    pass

    def predict_patch(self, patch, fs):
        """Yield Prediction on a patch, as of this Round.
        Uses estimators trained on examples of the arg FeatureSpec fs.
        """
        classifier = self.classifier
        estimators = self.trained_estimators()

        for feature in patch.features:
            if feature.spec.id == fs:
                return estimators[fs].predict(feature.nparray)[0]

            else:
                print(f"Breakage: {fs} not available for this patch!")

    def detect(self, blob):
        estimators = self.trained_estimators()
        for patch in blob.patches:
            for feature in patch.features:
                try:
                    value = estimators[feature.spec.id].predict(feature.nparray)
                    yield Prediction(
                        value=value, feature=feature, patch=patch, round=self
                    )
                except KeyError(e):
                    pass

    def subsample_estimator(
        self,
        subsample_ratio,
        cur_round_only=False,
        use_hard_negatives=False,
        featurespecs=None,
        trainval_split=False,
    ):
        """Train estimator for this Round to satisfy the subsample_ratio or
        only use hard negatives to satisfy the desired subsample_ratio.

        subsample_ratio: is so that we end up with n_negatives_train =
        subsample_ratio*n_positives_train, or the total number of
        negatives, whichever is less

        use_hard_negatives_fs: first train a randomly sampled estimator using
        subsample ratio, then retrain using hardest negatives, train with
        this feature spec
        """
        if cur_round_only:
            examples = self.examples
        else:
            examples = list(self.classifier.examples_upto(self))

        if subsample_ratio > 0:
            neg_ex = [i for i in examples if i.value == False]
            pos_ex = [i for i in examples if i.value == True]

            n_subsample = min(len(neg_ex), int(subsample_ratio * len(pos_ex)))

            subsample_indices = random.sample(range(len(neg_ex)), n_subsample)
            print(f"Number of subsampled negatives {len(subsample_indices)}")

            subexamples = pos_ex + [neg_ex[i] for i in subsample_indices]
            print(f"Subsampled training set size {len(subexamples)}")

            estimators, scores = self.trained_estimators(
                cur_round_only=subexamples,
                featurespecs=featurespecs,
                trainval_split=trainval_split,
                output_scores=True,
            )

            if use_hard_negatives:
                for fs in featurespecs:
                    # for each fs, find the hard negs in the original neg_ex
                    # (this may include training examples)
                    neg_samples = [
                        ex.patch.features.filter(Feature.spec_id == fs.id).one().vector
                        for ex in neg_ex
                    ]
                    predictions = estimators[fs.id].predict(neg_samples)
                    # retrain estimator using those as the neg_ex
                    hard_negative_example_ids = [
                        ind for ind, ex in enumerate(neg_ex) if predictions[ind] == 1
                    ]
                    estimators[f"hard_neg_ex_ids_{fs.id}"] = [
                        ex.id for ind, ex in enumerate(neg_ex) if predictions[ind] == 1
                    ]
                    print(
                        f"Number of hard negatives / total {len(hard_negative_example_ids)} / {len(neg_samples)}"
                    )

                    # New subsampled inds include hard negs plus a random selection to get to correct subsample_ratio
                    # if len(hard_negative_example_ids) >= len(subsample_indices):
                    #   subsample_indices = hard_negative_example_ids[:len(subsample_indices)]
                    # else:
                    #   subsample_indices = hard_negative_example_ids + subsample_indices[:len(subsample_indices)-len(hard_negative_example_ids)]
                    subsample_indices = list(
                        set(hard_negative_example_ids + subsample_indices)
                    )

                    print(
                        f"Number of subsampled negatives w/ hard negs {len(subsample_indices)}"
                    )

                    subexamples = pos_ex + [neg_ex[i] for i in subsample_indices]
                    print(
                        f"Subsampled training set size w/ hard negs {len(subexamples)}"
                    )
                    tmp_est, tmp_scores = self.trained_estimators(
                        cur_round_only=subexamples,
                        featurespecs=[fs],
                        trainval_split=trainval_split,
                        output_scores=True,
                    )
                    estimators[fs.id] = tmp_est[fs.id]
                    scores[fs.id] = tmp_scores[fs.id]

            return estimators, scores

        else:
            print("Round examples not updated, subsample_ration must be > 0")
            return []

    def trained_estimators(
        self,
        cur_round_only=False,
        featurespecs=None,
        trainval_split=False,
        output_scores=False,
    ):
        """Creates an estimator for each FeatureSpec used by the Dataset,
        then trains each on the features of known examples.

        cur_round_only: if True only use examples attached to this round,
        if False use examples from this and all previous rounds of this
        classifier. If cur_round_only is a list of examples, use those.

        featurespecs: features to use for training if different than
        featurespecs attached to self's dataset

        trainval_split: first split for train and val to get a val set
        score, still returns calssifier trained on whole dataset.
        80/20 split used.
        """
        scores = {}

        if featurespecs is None:
            featurespecs = self.classifier.dataset.featurespecs

        estimators = {}

        for fs in featurespecs:
            samples, results = ([], [])
            scores[fs.id] = {}

            if cur_round_only:
                if type(cur_round_only) == list:
                    examples = cur_round_only
                else:
                    examples = self.examples
            else:
                examples = self.classifier.examples_upto(self)

            for ex in examples:
                feat = Feature.of(patch=ex.patch, spec=fs)
                samples.append(feat.vector)
                results.append(1 if ex.value else -1)

            print(
                f"Training estimator with {len(samples)} examples for feature spec {fs} ..."
            )
            # Evaluate on train val split
            # TODO: this would be better if we did k-fold crossvalidation
            neg_indices = [i for i in range(len(results)) if results[i] != 1]
            pos_indices = [i for i in range(len(results)) if results[i] == 1]
            if trainval_split:
                print(
                    f"{len(pos_indices)} positive examples, {len(neg_indices)} negative examples"
                )

                # TODO make val set population balanced
                random.shuffle(neg_indices)
                random.shuffle(pos_indices)
                train_neg_len = int(0.8 * len(neg_indices))
                train_pos_len = int(0.8 * len(pos_indices))
                max_val_pop = min(
                    (len(pos_indices) - train_pos_len),
                    (len(neg_indices) - train_neg_len),
                )

                # Train estimator with natural population
                train_indices = (
                    neg_indices[:train_neg_len] + pos_indices[:train_pos_len]
                )
                train_samples = [samples[i] for i in train_indices]
                train_results = [results[i] for i in train_indices]

                # Validate with balanced population
                val_indices = (
                    neg_indices[train_neg_len:][:max_val_pop]
                    + pos_indices[train_pos_len:][:max_val_pop]
                )
                val_samples = [samples[i] for i in val_indices]
                val_results = [results[i] for i in val_indices]
                print(
                    f"{len(train_samples)} training examples, {len(val_samples)} validation examples"
                )

                estimators[fs.id] = self.classifier.estimator.instantiate()
                estimators[fs.id].fit(train_samples, train_results)
                val_score = estimators[fs.id].score(val_samples, val_results)
                if output_scores:
                    scores[fs.id]["val"] = val_score
                print(
                    "Validation Score for FeatureSpec {} (Chance 0.5): {:.5}".format(
                        fs.id, val_score
                    )
                )

            # Retrain using all examples
            estimators[fs.id] = self.classifier.estimator.instantiate()
            estimators[fs.id].fit(samples, results)
            train_score = estimators[fs.id].score(samples, results)
            if output_scores:
                scores[fs.id]["train"] = train_score

            print(
                "Final Training Score for FeatureSpec {} (Chance {:.5}): {:.5}".format(
                    fs.id,
                    len(pos_indices) / (len(pos_indices) + len(neg_indices)),
                    train_score,
                )
            )

        if output_scores:
            return estimators, scores
        else:
            return estimators

    def choose_queries(self):
        """Yield PatchQueries to ask humans about some Patches.  Currently,
        chooses the highest predictions, but might become more clever, to
        select Patches that are likely to most improve the Classifier.

        Even before that, should certainly be made clever enough to draw
        on high predictions from different FeatureSpecs.
        """
        if not hasattr(config, "active_query_strategy"):
            config.active_query_strategy = "most_confident"
        half_query_num = int(round(config.query_num / 2.0))
        predictions = []
        if config.active_query_strategy == "most_confident":
            predictions = self.predictions[: config.query_num]
        elif config.active_query_strategy == "least_confident":
            predictions = self.predictions[-config.query_num :]
        elif config.active_query_strategy == "hybrid":
            preds_most_confident = [p for p in self.predictions[: config.query_num]]
            preds_least_confident = [p for p in self.predictions[-config.query_num :]]
            preds_least_confident.reverse()
            # remove duplicates
            pred_map = collections.OrderedDict()
            for i in range(
                min(
                    config.query_num,
                    len(preds_most_confident),
                    len(preds_least_confident),
                )
            ):
                pm_el = preds_most_confident[i]
                pl_el = preds_least_confident[i]
                pred_map[pm_el.id] = pm_el
                pred_map[pl_el.id] = pl_el
            # pick the top query_num elements
            top_elements = list(pred_map.items())[: config.query_num]
            _, predictions = zip(*top_elements)
        else:
            print(
                f'Error: invalid active query strategy "{config.active_query_strategy}" specified'
            )
        for p in predictions:
            yield PatchQuery(predicted=p.value, patch=p.patch, round=self)

    def choose_distributed_queries(self):
        """Yield PatchQueries for debugging purposes. Uniformally sample
        from all predictions.
        """
        query_num = 1000
        stride = int(len(self.predictions.all()) / query_num)
        for p in self.predictions[::stride]:
            yield PatchQuery(predicted=p.value, patch=p.patch, round=self)

    def __repr__(self):
        return model_debug(self)

    def average_precision(self, keyword, add_neg_ratio=None):
        """Returns the AP of the Round's estimator on the elements of the
        keyword.
        Adds negatives randomly selected from the keyword's dataset
        up to a pos/neg ratio of <add_neg_ratio>.
        """

        estimators = self.trained_estimators()
        # get all patches from keyword
        y = {}
        feats = {}
        num_seeds = len(keyword.seeds.all())
        for ex in keyword.seeds:
            for feature in ex.patch.features:
                if feature.spec.id not in y.keys():
                    y[feature.spec.id] = {}
                    y[feature.spec.id]["true"] = []
                    feats[feature.spec.id] = []
                y[feature.spec.id]["true"].append(1.0 if ex.value else 0.0)
                feats[feature.spec.id].append(feature.vector)

                print("{} of {}".format(len(y[feature.spec.id]["true"]), num_seeds))
        # get additional negatives if need be
        if add_neg_ratio is not None:
            num_pos = len(keyword.seeds.filter(Example.value == True).all())
            num_neg = len(keyword.seeds.filter(Example.value == False).all())
            kw_patches = [ex.patch for ex in keyword.seeds]
            ds_blobs = keyword.dataset.blobs
            while np.true_divide(num_neg, (num_neg + num_pos)) < add_neg_ratio:
                blob = ds_blobs[random.randint(0, len(ds_blobs) - 1)]
                patches = blob.patches.order_by(func.random()).all()
                for patch in patches:
                    if patch not in kw_patches:
                        break
                for feature in patch.features:
                    y[feature.spec.id]["true"].append(0.0)
                    feats[feature.spec.id].append(feature.vector)
                num_neg += 1
                print("Number of added negatives {}".format(num_neg))
        # calculate AP
        ap = {}
        for ftype in y.keys():
            y[ftype]["pred"] = estimators[ftype].decision_function(
                np.asarray(feats[ftype])
            )
            ap[ftype] = average_precision_score(y[ftype]["true"], y[ftype]["pred"])
        return y


class Patch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # TODO: test
    x = db.Column(db.Integer, nullable=False)  # , CheckConstraint('x > 0'))
    y = db.Column(db.Integer, nullable=False)  # , CheckConstraint('y > 0'))
    width = db.Column(db.Integer, db.CheckConstraint("width>0"), nullable=False)
    height = db.Column(db.Integer, db.CheckConstraint("height>0"), nullable=False)
    fliplr = db.Column(db.Boolean, nullable=False, default=False)
    rotation = db.Column(db.Float, nullable=False, default=0.0)

    blob_id = db.Column(db.Integer, db.ForeignKey("blob.id"), index=True)
    blob = db.relationship("Blob", backref=db.backref("patches", lazy="dynamic"))

    # for creating video patches
    frame_rate = db.Column(db.Float, nullable=False, default=0.0)

    @property
    def is_video(self):
        return self.frame_rate > 0.0

    @property
    def size(self):
        assert self.width == self.height
        return self.width

    @property
    def bbox(self):
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def ensure(model, **kwargs):
        existing = model.query.filter_by(**kwargs).first()
        if existing:
            return existing
        return model(**kwargs)

    @classmethod
    def ifNew(cls, **kwargs):
        if not cls.query.filter_by(**kwargs).first():
            return cls(**kwargs)

    def materialize(self):
        assert self.id
        ext = self.blob.ext
        filename = "patch-" + str(self.id) + ext
        complete = cache_fname(config.CACHE_DIR, filename)

        if not os.path.exists(complete):
            imageio.imwrite(complete, self.image)

        return complete

    @property
    def image(self):
        img = self.blob.image
        if img == []:
            return []
        if self.fliplr:
            img = np.fliplr(img)
        if self.rotation != 0.0:
            img = rotate(img, self.rotation)

        # print("image shape {}: (x,y,w,h) {} ".format(img.shape, (self.x, self.y, self.width, self.height)))

        crop = img[self.y : self.y + self.height, self.x : self.x + self.width]

        # Remove alpha channel if present
        try:
            if crop.shape[2] == 4:
                crop = crop[:, :, 0:3]
        # Replicate channels if image is Black and White
        except IndexError as e:
            tmp = np.zeros((crop.shape[0], crop.shape[1], 3))
            tmp[:, :, 0] = crop
            tmp[:, :, 1] = crop
            tmp[:, :, 2] = crop
            crop = tmp

        return crop

    @property
    def video(self):
        """This applies a crop to each frame of the video, but doesn't change the
        number of frames or color resolution"""
        if not self.is_video:
            return None
        vid = self.blob.video
        if self.fliplr or self.rotation != 0.0:
            print(
                """Warning: video patches don't yet support fliplr or rotation params.
      No changes were applied."""
            )
        # video is of the shape [L, H, W, 3 (RGB)]
        vid_length = vid.shape[0]
        assert vid.shape == (vid_length, self.height, self.width, 3)
        crop = vid[:, self.y : self.y + self.height, self.x : self.x + self.width, :3]
        return crop

    @property
    def data(self):
        return self.video if self.is_video else self.image

    @property
    def url(self):
        return url_for("patch", id=self.id)

    def __repr__(self):
        return model_debug(self)


class Feature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # For storing image (vector) and video (array) features as multidimensional arrays
    vector = db.Column(types.JSON)  # postgresql.ARRAY(db.Float))
    array = db.Column(types.JSON)  # postgresql.ARRAY(db.Float, dimensions=2))

    patch_id = db.Column(db.Integer, db.ForeignKey("patch.id"), index=True)
    patch = db.relationship("Patch", backref=db.backref("features", lazy="dynamic"))

    spec_id = db.Column(db.Integer, db.ForeignKey("feature_spec.id"), index=True)
    spec = db.relationship(
        "FeatureSpec", backref=db.backref("features", lazy="dynamic")
    )

    # for creating video features
    frame_rate = db.Column(db.Float, nullable=False, default=0.0)

    def __init__(self, data=None, **kwargs):
        # when this object is loaded from the db, the data comes in as lists
        # so convert here for consistency
        if data is not None:
            if isinstance(data, np.ndarray):
                data = data.tolist()
        if "array" in kwargs:
            if isinstance(kwargs["array"], np.ndarray):
                kwargs["array"] = kwargs["array"].tolist()
        if "vector" in kwargs:
            if isinstance(kwargs["vector"], np.ndarray):
                kwargs["vector"] = kwargs["vector"].tolist()

        self.vector = []
        self.array = []
        super(Feature, self).__init__(**kwargs)
        if self.patch.is_video:
            # TODO have this be updated this based on the feature spec computation, if there's any downsampling
            self.frame_rate = self.patch.frame_rate
        if data is None:
            return

        if self.is_video:
            self.array = data
        else:
            self.vector = data

    @property
    def is_video(self):
        return self.frame_rate > 0.0

    # TODO: this should be true, but not working now...
    # __table_args__ = (db.UniqueConstraint('patch_id', 'spec_code', name='_patch_spec_comb'),)

    @classmethod
    def ifNew(cls, **kwargs):
        if not cls.query.filter_by(**kwargs).first():
            return cls(**kwargs)

    @classmethod
    def of(cls, **kwargs):
        return cls.query.filter_by(**kwargs).one()

    def closest(self):
        best = (None, float("inf"))
        for f in self.spec.features:
            if f.id == self.id:
                break
            sqd = self.squared_distance(f)
            if sqd < best[1]:
                best = (f, sqd)
        return best

    @staticmethod
    def _squared_distance(vec1, vec2):
        return sum([(a - b) ** 2 for a, b in zip(vec1, vec2)])

    def squared_distance(self, other):
        if self.is_video:
            array = np.array(self.array)
            other_array = np.array(other.array)
            assert (
                array.shape == other_array.shape
            ), "Video arrays do not have matching dimensions"
            return self._squared_distance(
                array.reshape(1, -1), other_array.reshape(1, -1)
            )
        else:
            return self._squared_distance(self.vector, other.vector)

    def distance(self, other):
        return np.sqrt(self.squared_distance(other))

    @property
    def nparray(self):
        if self.is_video:
            return np.array(self.array)
        else:
            return np.array(self.vector).reshape(1, -1)

    def __repr__(self):
        return (
            "Feature#"
            + str(self.id)
            + ":"
            + str(self.spec.name)
            + " "
            + str(self.vector)
        )


class Example(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Boolean, nullable=False)

    patch_id = db.Column(
        db.Integer, db.ForeignKey("patch.id"), index=True, nullable=False
    )
    patch = db.relationship("Patch", backref=db.backref("examples", lazy="dynamic"))

    # An example is associated with a keyword OR a round of classifier training
    keyword_id = db.Column(db.Integer, db.ForeignKey("keyword.id"), nullable=True)
    keyword = db.relationship("Keyword", backref=db.backref("seeds", lazy="dynamic"))

    round_id = db.Column(db.Integer, db.ForeignKey("round.id"), nullable=True)
    round = db.relationship("Round", backref=db.backref("examples", lazy="dynamic"))

    def __repr__(self):
        return model_debug(self)

    @classmethod
    def ifNew(model, **kwargs):
        if not model.query.filter_by(**kwargs).first():
            return model(**kwargs)


class Prediction(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    value = db.Column(db.Float, index=True, nullable=False)

    feature_id = db.Column(db.Integer, db.ForeignKey("feature.id"), index=True)
    feature = db.relationship(
        "Feature", backref=db.backref("predictions", lazy="dynamic")
    )

    patch_id = db.Column(db.Integer, db.ForeignKey("patch.id"), index=True)
    patch = db.relationship("Patch", backref=db.backref("predictions", lazy="dynamic"))

    round_id = db.Column(db.Integer, db.ForeignKey("round.id"), nullable=False)
    round = db.relationship(
        "Round",
        backref=db.backref(
            "predictions", order_by="Prediction.value.desc()", lazy="dynamic"
        ),
    )

    def __repr__(self):
        return model_debug(self)


# This is identical to the Prediction Object, but intended only to be used
# for test and validation purposes, not for active query.
class ValPrediction(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    value = db.Column(db.Float, index=True, nullable=False)

    feature_id = db.Column(db.Integer, db.ForeignKey("feature.id"), index=True)
    feature = db.relationship(
        "Feature", backref=db.backref("valpredictions", lazy="dynamic")
    )

    patch_id = db.Column(db.Integer, db.ForeignKey("patch.id"), index=True)
    patch = db.relationship(
        "Patch", backref=db.backref("valpredictions", lazy="dynamic")
    )

    round_id = db.Column(db.Integer, db.ForeignKey("round.id"), nullable=False)
    round = db.relationship(
        "Round",
        backref=db.backref(
            "valpredictions", order_by="ValPrediction.value.desc()", lazy="dynamic"
        ),
    )

    def __repr__(self):
        return model_debug(self)


class PatchQuery(db.Model):
    """A Classifier thinks it would be a good idea to ask about a patch."""

    id = db.Column(db.Integer, primary_key=True)
    predicted = db.Column(db.Float, nullable=False)

    patch_id = db.Column(db.Integer, db.ForeignKey("patch.id"), nullable=False)
    patch = db.relationship("Patch", backref=db.backref("queries", lazy="dynamic"))

    round_id = db.Column(db.Integer, db.ForeignKey("round.id"), nullable=False)
    round = db.relationship(
        "Round",
        backref=db.backref(
            "queries", order_by="PatchQuery.predicted.desc()", lazy="dynamic"
        ),
    )

    def __repr__(self):
        return model_debug(self)


class PatchResponse(db.Model):
    """A human has answered a PatchQuery (as part of a HitResponse)"""

    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Boolean, nullable=False)

    query_id = db.Column(db.Integer, db.ForeignKey("patch_query.id"), nullable=False)
    patchquery = db.relationship(
        "PatchQuery", backref=db.backref("responses", lazy="dynamic")
    )

    hit_id = db.Column(db.Integer, db.ForeignKey("hit_response.id"), nullable=False)
    hitresponse = db.relationship(
        "HitResponse", backref=db.backref("patch_responses", lazy="dynamic")
    )

    def __repr__(self):
        return model_debug(self)


class HitResponse(db.Model):
    """A set of PatchResponses, all done by a user in one HIT"""

    id = db.Column(db.Integer, primary_key=True)
    # the completion time for all of the patch responses associated with this HIT
    time = db.Column(db.Float(), nullable=False)
    # the (self-reported) confidence of the labeling user
    confidence = db.Column(db.Integer)

    # the user that labeled the associated PatchResponses
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship("User", backref=db.backref("hits", lazy="dynamic"))

    def __repr__(self):
        return model_debug(self)


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    blob_id = db.Column(db.Integer, db.ForeignKey("blob.id"), index=True)
    blob = db.relationship("Blob", backref=db.backref("detections", lazy="dynamic"))

    @property
    def url(self):
        return url_for("detect", id=self.id)

    def __repr__(self):
        return model_debug(self)


# Convenience routine simple repr implementation of models.
def model_debug(m):
    id = m.id
    c = dict.copy(m.__dict__)
    del c["_sa_instance_state"]
    if "id" in c.keys():
        del c["id"]
    return type(m).__name__ + "#" + str(id) + ":" + str(c)


def cache_fname(cache_dir, fname):
    hash_dir = os.path.join(cache_dir, "{}".format(hash(fname) % 10**8))
    if not os.path.exists(hash_dir):
        os.mkdir(hash_dir)
    return os.path.join(hash_dir, fname)


# This adapts (1-dimensional) numpy arrays to Postgres
# We should make it do n-dimensionals eventually.
from psycopg2.extensions import register_adapter, AsIs


def adapt_numpy(np):
    return AsIs(",".join([str(f) for f in np]))


register_adapter(np.ndarray, adapt_numpy)
