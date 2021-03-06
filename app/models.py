#!/usr/bin/env python
import resource
import random
import importlib
import os, time
import urllib


import psutil
from flask import url_for
from flask_user import UserManager, UserMixin, SQLAlchemyAdapter, current_user
from more_itertools import chunked
from scipy import misc
from scipy.ndimage.interpolation import rotate
from sklearn.metrics import average_precision_score
from  sqlalchemy.sql.expression import func
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql
import boto3
import exif
import numpy as np

from app import app, db
import config
import tasks
import extract

s3 = boto3.resource('s3')

# TODO: Consider this for easier form creation from existing models.
# https://wtforms-alchemy.readthedocs.org/en/latest/index.html
# from wtforms_alchemy import ModelForm


# Very basic. Consider Flask-User before getting more complex.
class User(db.Model, UserMixin):
  id = db.Column(db.Integer, primary_key = True)
  username = db.Column(db.String, nullable = False, unique = True)
  # if password is null, then can't login in, except by mturk-like bypass
  password = db.Column(db.String, nullable = True)
  reset_password_token = db.Column(db.String, nullable= True)

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

dataset_x_blob = db.Table('dataset_x_blob', db.Model.metadata,
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id')),
    db.Column('blob_id', db.Integer, db.ForeignKey('blob.id'))
)

def s3_url(location):
  assert location[:5] == 's3://'
  (bucket, key) = location[5:].split("/")
  return "http://%s.s3.amazonaws.com/%s" % (bucket, key)

def static_url(location):
  prefix = config.kairoot + '/app/static/'
  assert location.startswith(prefix)
  return url_for('static', filename=location[len(prefix):])



def clean_cache(s):
  dir = config.CACHE_DIR
  now = time.time()
  for fn in os.listdir(dir):
    complete = os.path.join(dir, fn)
    last_access = os.path.getatime(complete)
    if now - last_access > 24 * 60 * 60:
      os.remove(complete)


class Blob(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  ext = db.Column(db.String)
  mime = db.Column(db.String)
  location = db.Column(db.String)
  latitude = db.Column(db.Float)
  longitude = db.Column(db.Float)

  URL_MAP = {
    config.kairoot + '/app/static/' : static_url,
    's3://' : s3_url,
  }

  def __init__(self, location):
    self.location = location
    self.ext = os.path.splitext(location)[1]
    self.latitude, self.longitude = (None, None)
    if self.ext == '.jpg':
      self.mime = 'image/jpeg'
    else:
      self.mime = 'image/'+self.ext.replace('.', '')
    if self.local:
      self.latitude, self.longitude = self.read_lat_lon()
    self.img = None

  @orm.reconstructor
  def init_on_load(self):
    self.img = None

  def open(self):
    return open(self.materialize(), "rb")

  def materialize(self):
    if self.local:
      return self.location

    assert self.id
    complete = os.path.join(config.CACHE_DIR, self.filename)

    if not os.path.exists(complete):
      urllib.urlretrieve(self.url, complete)
    return complete

  def read_lat_lon(self):
    return exif.get_lat_lon(exif.get_data(self.materialize()))

  @property
  def features(self):
    for patch in self.patches:
      for feature in patch:
        yield feature

  @property
  def image(self):
    #if self.img is not None:
    #  return self.img
    try:
      with self.open() as f:
        img = misc.imread(f)
      return img
    except IOError, e:
      print 'Could not open image file for {}'.format(self)
      return []

  def reset(self):
    self.img = None

  @property
  def local(self):
    return self.location[0] == '/' and os.path.exists(self.location)
  @property
  def on_s3(self):
    return self.location.startswith("s3://")

  @property
  def url(self):
    url = self.location
    for prefix, change in Blob.URL_MAP.iteritems():
      if url.startswith(prefix):
        return change(url)
    return url

  @property
  def filename(self):
    return 'blob-'+str(self.id)+self.ext

  BUCKET = config.APPNAME+'-blobs'
  def migrate_to_s3(self):
    if self.on_s3:
      return
    with self.open() as body:
      s3.Bucket(Blob.BUCKET).put_object(Key=self.filename, Body=body)
    self.location = "s3://%s/%s" % (Blob.BUCKET, self.filename)

  def __repr__(self):
    return "Blob#"+str(self.id)+":"+self.location

dataset_x_patchspec = db.Table('dataset_x_patchspec', db.Model.metadata,
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id')),
    db.Column('patchspec_id', db.Integer, db.ForeignKey('patch_spec.id'))
)

class PatchSpec(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  # Starting size (smallest patches)
  width = db.Column(db.Integer, nullable=False)
  height = db.Column(db.Integer, nullable=False)

  # Fraction of patch to keep while sliding over for next patch
  xoverlap = db.Column(db.Float, nullable=False) # 0 < xoverlap <= 1
  # Fraction of patch to keep while sliding down for next row of patches
  yoverlap = db.Column(db.Float, nullable=False) # 0 < yoverlap <= 1

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
    return "/patchspec/"+str(self.id)

  def __repr__(self):
    upto = " by "+str(self.scale)
    if self.xoverlap == self.yoverlap:
      overlap = str(self.xoverlap) + " overlap"
    else:
      overlap = str(self.yoverlap) + "/" + str(self.xoverlap) +" overlap"
    return self.name + ": " + str(self.height) + "x" + str(self.width) \
      + upto + " with " + overlap

  def create_blob_patches(self, blob):
    print 'Creating patches for {}'.format(blob)
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    img = blob.image

    w = self.width
    h = self.height
    pind = 0
    try:
      while w < img.shape[1] and h < img.shape[0]:
        for patch in self.create_sized_blob_patches(blob, img.shape, w, h):
          pind += 1
          yield patch
        w = int(w * self.scale)
        h = int(h * self.scale)
    except IndexError, e:
      print 'Index Error for {}'.format(blob)
      print 'Error on Patch #{}'.format(pind)
      print 'Blob\'s Image Shape: {}'.format(img.shape)
      print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      blob.reset()
      print '*** Reseting blob image ***'
      print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      return

  def create_sized_blob_patches(self, blob, size, width, height):
    # How far each slide will go
    xstep = int(width * (1-self.xoverlap))
    ystep = int(height * (1-self.yoverlap))

    # The available room for sliding
    xdelta = size[1]-width
    ydelta = size[0]-height

    # How many slides, and how much unused "slide space"
    xsteps, extra_width = divmod(xdelta, xstep)
    ysteps, extra_height = divmod(ydelta, ystep)

    # Divy up the unused slide space, to lose evenly at edges
    left_indent = extra_width // 2
    top_indent = extra_height // 2

    for x in xrange(left_indent, xdelta, xstep):
      for y in xrange(top_indent, ydelta, ystep):
        yield Patch.ensure(blob=blob, x=x, y=y,
                           width=width, height=height, fliplr=False)
        if not self.fliplr:
          continue
        yield Patch.ensure(blob=blob, x=x, y=y,
                           width=width, height=height, fliplr=True)

dataset_x_featurespec = db.Table('dataset_x_featurespec', db.Model.metadata,
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id')),
    db.Column('featurespec_id', db.Integer, db.ForeignKey('feature_spec.id'))
)


class FeatureSpec(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)
  cls = db.Column(db.String, nullable = False)
  params = db.Column(postgresql.JSON)

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
      self.instance = eval("importlib.import_module('%s').%s()" %
                           (module, classname))
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
    return "/featurespec/"+str(self.id)


  def analyze_blob(self, blob):
    # TODO: this could be a globally set var that shares with CNN obj
    batch_size = 500
    iter = 0
    for patches in chunked(self.undone_patches(blob), batch_size):
      print 'calculating {}x500 patch features for {}'.format(iter, blob)
      iter += 1
      imgs = [p.image for p in patches]
      feats = self.instance.extract_many(imgs)
      assert len(patches) == len(feats), 'The number of patches and features are different for {}'.format(blob)
      for idx, f in enumerate(feats):
        yield Feature(patch=patches[idx], spec=self,
                      vector=f)

  def undone_patches(self, blob):
    for p in blob.patches:
      if Feature.query.filter_by(patch=p, spec=self).count() == 0:
        yield p


  def analyze_patch(self, patch):
    if Feature.query.filter_by(patch=patch, spec=self).count() > 0:
      return None
    return Feature(patch=patch, spec=self,
                   vector=self.instance.extract(patch.image))

class Dataset(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
  owner = db.relationship('User', backref = db.backref('datasets', lazy = 'dynamic'))

  blobs = db.relationship("Blob", secondary=dataset_x_blob)
  patchspecs = db.relationship("PatchSpec",
                               secondary=dataset_x_patchspec,
                               backref="datasets")
  featurespecs = db.relationship("FeatureSpec",
                                 secondary=dataset_x_featurespec,
                                 backref="datasets")

  def __init__(self, **kwargs):
      super(Dataset, self).__init__(**kwargs)
      if self.owner is None and current_user.is_authenticated:
        self.owner = current_user

  def create_blob_features(self, blob):
    print 'calculating features for {}'.format(blob)
    batch_size = 500
    for ps in self.patchspecs:
      print ps

      for patches in chunked(ps.create_blob_patches(blob), batch_size):
        for p in patches:
          if p:
            db.session.add(p)
        db.session.commit()
        print p

    for fs in self.featurespecs:
      print fs
      feats = fs.analyze_blob(blob)
      for feat in chunked(feats, batch_size):
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for f in feat:
          db.session.add(f)
        db.session.commit()
        if fs.instance.__class__ is extract.CNN:
          fs.instance.del_networks()
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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


class Keyword(db.Model):
  '''
  Keywords can be created individually or be asscociated with a dataset
  When associated with a dataset, the keyword will have a defn_file that contains
  the location of a csv listing the images, patch locations, and label values of the
  examples associated with this keyword.
  '''
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
  owner = db.relationship('User', backref = db.backref('keywords', lazy = 'dynamic'))

  geoquery_id = db.Column(db.Integer, db.ForeignKey('geo_query.id'))
  geoquery = db.relationship('GeoQuery', backref = db.backref('keywords', lazy = 'dynamic'))

  defn_file = db.Column(db.String)
  dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
  dataset = db.relationship('Dataset', backref = db.backref('keywords', lazy = 'dynamic'))

  def __init__(self, **kwargs):
      super(Keyword, self).__init__(**kwargs)
      if self.owner is None and current_user.is_authenticated:
        self.owner = current_user


  @property
  def url(self):
    return url_for("keyword", id=self.id)

  def __repr__(self):
    return model_debug(self)

class GeoQuery(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  @property
  def url(self):
    return "/geo/"+self.name

  def __repr__(self):
    return model_debug(self)

estimators = [
  'sklearn.neighbors.KNeighborsRegressor',
  'sklearn.neighbors.RadiusNeighborsRegressor',
  'sklearn.linear_model.LinearRegression'
]

class Estimator(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  cls = db.Column(db.String, nullable = False)
  params = db.Column(postgresql.JSON)

  @classmethod
  def ifNew(model, **kwargs):
      if not model.query.filter_by(**kwargs).first():
          return model(**kwargs)

  def instantiate(self):
      parts = self.cls.split(".")
      module = ".".join(parts[:-1])
      classname = parts[-1]
      instance = eval("importlib.import_module('%s').%s()" %
                      (module, classname))
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

  id = db.Column(db.Integer, primary_key = True)

  owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
  owner = db.relationship('User', backref = db.backref('classifiers', lazy = 'dynamic'))

  dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
  dataset = db.relationship('Dataset', backref = db.backref('classifiers', lazy = 'dynamic'))

  keyword_id = db.Column(db.Integer, db.ForeignKey('keyword.id'), nullable=False)
  keyword = db.relationship('Keyword', backref = db.backref('classifiers', lazy = 'dynamic'))

  estimator_id = db.Column(db.Integer, db.ForeignKey('estimator.id'), nullable=False)
  estimator = db.relationship('Estimator')

  def __init__(self, **kwargs):
      super(Classifier, self).__init__(**kwargs)
      if self.owner is None and current_user.is_authenticated:
        self.owner = current_user
      zero = Round(classifier = self)

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


class Round(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  number = db.Column(db.Integer, nullable = False, default = 0)

  classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'), nullable=False)
  classifier = db.relationship('Classifier',
                               backref = db.backref('rounds',
                                                    order_by='Round.number',
                                                    lazy = 'dynamic'))
  def predict(self, ds=None):
    '''Yield Predictions for all Patches in the Dataset, as of this Round.
    Uses estimators trained on examples for each FeatureSpec to make
    predictions against its Features.
    '''
    classifier = self.classifier
    estimators = self.trained_estimators()

    if ds is None:
      ds = classifier.dataset
    for blob in ds.blobs:
      for patch in blob.patches:
        for feature in patch.features:
          #TODO: I don't think we want predict here, want decision function or probability
          # value = estimators[feature.spec.id].predict(feature.vector)
          value = estimators[feature.spec.id].decision_function(feature.vector)
          yield Prediction(value=value, feature=feature, round=self)

  def predict_patch(self, patch, fs):
    '''Yield Prediction on a patch, as of this Round.
    Uses estimators trained on examples of the arg FeatureSpec fs.
    '''
    classifier = self.classifier
    estimators = self.trained_estimators()

    for feature in patch.features:
      if feature.spec.id == fs:
        return estimators[fs].predict(feature.vector)[0]

  def detect(self, blob):
    estimators = self.trained_estimators()
    for patch in blob.patches:
      for feature in patch.features:
        value = estimators[feature.spec.id].predict(feature.vector)
        yield Prediction(value=value, feature=feature, round=self)

  def trained_estimators(self):
    '''Creates an estimator for each FeatureSpec used by the Dataset,
    then trains each on the features of known examples.
    '''
    estimators = {}
    for fs in self.classifier.dataset.featurespecs:
      samples, results = ([], [])
      for ex in self.classifier.examples_upto(self):
        feat = Feature.of(patch=ex.patch, spec=fs)
        samples.append(feat.vector)
        results.append(1 if ex.value else -1)

      estimators[fs.id] = self.classifier.estimator.instantiate()
      estimators[fs.id].fit(samples, results)
    return estimators

  def choose_queries(self):
    '''Yield PatchQueries to ask humans about some Patches.  Currently,
    chooses the highest predictions, but might become more clever, to
    select Patches that are likely to most improve the Classifier.

    Even before that, should certainly be made clever enough to draw
    on high predictions from different FeatureSpecs.
    '''
    for p in self.predictions[:config.query_num]:
      yield PatchQuery(predicted=p.value, patch=p.feature.patch, round=self)


    def __repr__(self):
      return model_debug(self)

  def average_precision(self, keyword, add_neg_ratio=None):
    '''Returns the AP of the Round's estimator on the elements of the
    keyword.
    Adds negatives randomly selected from the keyword's dataset
    up to a pos/neg ratio of <add_neg_ratio>.
    '''

    estimators = self.trained_estimators()
    # get all patches from keyword
    y = {}
    feats = {}
    num_seeds = len(keyword.seeds.all())
    for ex in keyword.seeds:
      for feature in ex.patch.features:
        if feature.spec.id not in y.keys():
          y[feature.spec.id] = {}
          y[feature.spec.id]['true'] = []
          feats[feature.spec.id] = []
        y[feature.spec.id]['true'].append( 1.0 if ex.value else 0.0 )
        feats[feature.spec.id].append(feature.vector)

        print '{} of {}'.format(len(y[feature.spec.id]['true']),
                                                  num_seeds)
    # get additional negatives if need be
    if add_neg_ratio is not None:
      num_pos = len(keyword.seeds.filter(Example.value == True).all())
      num_neg = len(keyword.seeds.filter(Example.value == False).all())
      kw_patches = [ex.patch for ex in keyword.seeds]
      ds_blobs = keyword.dataset.blobs
      while np.true_divide(num_neg, (num_neg+num_pos)) < add_neg_ratio:
        blob = ds_blobs[random.randint(0,len(ds_blobs)-1)]
        patches = blob.patches.order_by(func.random()).all()
        for patch in patches:
          if patch not in kw_patches:
            break
        for feature in patch.features:
          y[feature.spec.id]['true'].append( 0.0 )
          feats[feature.spec.id].append(feature.vector)
        num_neg += 1
        print 'Number of added negatives {}'.format(num_neg)
    # calculate AP
    ap = {}
    for ftype in y.keys():
      y[ftype]['pred'] = estimators[ftype].decision_function(np.asarray(feats[ftype]))
      ap[ftype] = average_precision_score(y[ftype]['true'], y[ftype]['pred'])
    return y

class Patch(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  x = db.Column(db.Integer, nullable = False)
  y = db.Column(db.Integer, nullable = False)
  width = db.Column(db.Integer, db.CheckConstraint('width>0'),
                    nullable = False)
  height = db.Column(db.Integer, db.CheckConstraint('height>0'),
                     nullable = False)
  fliplr = db.Column(db.Boolean, nullable=False, default=False)
  rotation = db.Column(db.Float, nullable=False, default=0.0)

  blob_id = db.Column(db.Integer, db.ForeignKey('blob.id'), index = True)
  blob = db.relationship('Blob', backref = db.backref('patches', lazy = 'dynamic'))

  @property
  def size(self):
    assert self.width == self.height
    return self.width

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
    filename = "patch-"+str(self.id)+ext;
    complete = os.path.join(config.CACHE_DIR, filename)

    if not os.path.exists(complete):
        print filename
        print psutil.virtual_memory()
        misc.imsave(complete, self.image)
        print psutil.virtual_memory()
        print "saved."
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

    print "image shape: "
    print img.shape
    print (self.y, self.y+self.height, self.x, self.x+self.width)

    crop = img[self.y:self.y+self.height, self.x:self.x+self.width]

    # Remove alpha channel if present
    try :
      if crop.shape[2] == 4:
        crop = crop[:,:,0:3]
    # Replicate channels if image is Black and White
    except IndexError, e:
      tmp = np.zeros((crop.shape[0], crop.shape[1], 3))
      tmp[:,:,0] = crop
      tmp[:,:,1] = crop
      tmp[:,:,2] = crop
      crop = tmp

    return crop

  @property
  def url(self):
    return url_for("patch", id=self.id)

  def __repr__(self):
    return model_debug(self)


class Feature(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  vector = db.Column(postgresql.ARRAY(db.Float), nullable=False)

  patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True)
  patch = db.relationship('Patch', backref = db.backref('features', lazy = 'dynamic'))

  spec_id = db.Column(db.Integer, db.ForeignKey('feature_spec.id'), index = True)
  spec = db.relationship('FeatureSpec', backref = db.backref('features', lazy = 'dynamic'))

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

  def squared_distance(self, other):
    return sum([(a-b)**2 for a,b in zip(self.vector, other.vector)])

  def distance(self, other):
    return sqrt(self.squared_distance(other))

  def __repr__(self):
    return "Feature#"+str(self.id)+":"+str(self.spec.name) + ' ' + str(self.vector)

class Example(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  value = db.Column(db.Boolean, nullable=False)

  patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), index = True, nullable=False)
  patch = db.relationship('Patch', backref = db.backref('examples', lazy = 'dynamic'))

  # An example is associated with a keyword OR a round of classifier training
  keyword_id = db.Column(db.Integer, db.ForeignKey('keyword.id'), nullable=True)
  keyword = db.relationship('Keyword', backref = db.backref('seeds', lazy = 'dynamic'))

  round_id = db.Column(db.Integer, db.ForeignKey('round.id'), nullable=True)
  round = db.relationship('Round', backref = db.backref('examples', lazy = 'dynamic'))

  def __repr__(self):
    return model_debug(self)


class Prediction(db.Model):
  id = db.Column(db.BigInteger, primary_key = True)
  value = db.Column(db.Float, index = True, nullable=False)

  feature_id = db.Column(db.Integer, db.ForeignKey('feature.id'), index = True, nullable=False)
  feature = db.relationship('Feature', backref = db.backref('predictions', lazy = 'dynamic'))

  round_id = db.Column(db.Integer, db.ForeignKey('round.id'), nullable=False)
  round = db.relationship('Round',
                          backref = db.backref('predictions',
                                               order_by='Prediction.value.desc()',
                                               lazy = 'dynamic'))
  def __repr__(self):
    return model_debug(self)

class PatchQuery(db.Model):
  """A Classifier thinks it would be a good idea to ask about a patch."""
  id = db.Column(db.Integer, primary_key = True)
  predicted = db.Column(db.Float, nullable=False)

  patch_id = db.Column(db.Integer, db.ForeignKey('patch.id'), nullable=False)
  patch = db.relationship('Patch', backref = db.backref('queries', lazy = 'dynamic'))

  round_id = db.Column(db.Integer, db.ForeignKey('round.id'), nullable=False)
  round = db.relationship('Round',
                          backref = db.backref('queries',
                                               order_by='PatchQuery.predicted.desc()',
                                               lazy = 'dynamic'))

  def __repr__(self):
    return model_debug(self)

class PatchResponse(db.Model):
  """A human has answered a PatchQuery (as part of a HitResponse)"""
  id = db.Column(db.Integer, primary_key = True)
  value = db.Column(db.Boolean, nullable=False)

  query_id = db.Column(db.Integer, db.ForeignKey('patch_query.id'), nullable=False)
  patchquery = db.relationship('PatchQuery', backref = db.backref('responses', lazy = 'dynamic'))

  hit_id = db.Column(db.Integer, db.ForeignKey('hit_response.id'), nullable=False)
  hitresponse = db.relationship('HitResponse', backref = db.backref('patch_responses', lazy = 'dynamic'))

  def __repr__(self):
    return model_debug(self)


class HitResponse(db.Model):
  """A set of PatchResponses, all done by a user in one HIT"""
  id = db.Column(db.Integer, primary_key = True)
  # the completion time for all of the patch responses associated with this HIT
  time = db.Column(db.Float(), nullable=False)
  # the (self-reported) confidence of the labeling user
  confidence = db.Column(db.Integer)

  # the user that labeled the associated PatchResponses
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
  user = db.relationship('User', backref = db.backref('hits', lazy = 'dynamic'))

  def __repr__(self):
    return model_debug(self)

class Detection(db.Model):
  id = db.Column(db.Integer, primary_key = True)

  blob_id = db.Column(db.Integer, db.ForeignKey('blob.id'), index = True)
  blob = db.relationship('Blob', backref = db.backref('detections', lazy = 'dynamic'))

  @property
  def url(self):
    return url_for("detect", id=self.id)

  def __repr__(self):
    return model_debug(self)

# Convenience routine simple repr implementation of models.
def model_debug(m):
  id = m.id
  c = dict.copy(m.__dict__)
  del c['_sa_instance_state']
  if 'id' in c.keys():
    del c['id']
  return type(m).__name__+"#"+str(id)+":"+str(c)

# This adapts (1-dimensional) numpy arrays to Postgres
# We should make it do n-dimensionals eventually.
from psycopg2.extensions import register_adapter, AsIs
def adapt_numpy(np):
    return AsIs(",".join([str(f) for f in np]))
register_adapter(np.ndarray, adapt_numpy)
