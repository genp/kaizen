#!/usr/bin/env python
from app import app, db

import boto3
import config
import os, time
import numpy as np
import importlib
import urllib
from PIL import Image
from scipy import misc
from scipy.ndimage.interpolation import rotate
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql
from flask_user import UserManager, UserMixin, SQLAlchemyAdapter

import exif
import tasks

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

user_manager = UserManager(SQLAlchemyAdapter(db, User), app)


dataset_x_blob = db.Table('dataset_x_blob', db.Model.metadata,
    db.Column('dataset_id', db.Integer, db.ForeignKey('dataset.id')),
    db.Column('blob_id', db.Integer, db.ForeignKey('blob.id'))
)

def s3_url(location):
  assert location[:5] == 's3://'
  (bucket, key) = location[5:].split("/")
  return "http://%s.s3.amazonaws.com/%s" % (bucket, key)

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
    config.clroot + '/app/static/' : 'http://localhost:8080/',
    's3://' : s3_url,
  }

  def __init__(self, location):
    self.location = location
    self.ext = os.path.splitext(location)[1]
    if self.ext == '.jpg':
      self.mime = 'image/jpeg'
      self.latitude, self.longitude = self.read_lat_lon()
    else:
      self.mime = 'image/'+self.ext.replace('.', '')
      self.latitude, self.longitude = (None, None)


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
        if hasattr(change, '__call__'):
          return change(url)
        return url.replace(prefix, change, 1)
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
    return "Blob:"+self.location

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

  def count_blob_patches(self, blob):
    img = Image.open(blob.materialize())

    w = self.width
    h = self.height
    sum = 0
    while w < img.size[0] and h < img.size[1]:
      sum += self.count_sized_blob_patches(blob, img.size, w, h)
      w = int(w * self.scale)
      h = int(h * self.scale)
    return sum

  def count_sized_blob_patches(self, blob, size, width, height):
    # How far each slide will go
    xstep = int(self.xoverlap * width)
    ystep = int(self.yoverlap * height)

    # The available room for sliding
    xdelta = size[0]-width
    ydelta = size[1]-height

    # How many slides, and how much unused "slide space"
    xsteps, extra_width = divmod(xdelta, xstep)
    ysteps, extra_height = divmod(ydelta, ystep)
    return (1+xsteps) * (1+ysteps)


  def create_dataset_patches(self, ds):
    for blob in ds.blobs:
      for patch in self.create_blob_patches(blob):
        yield patch

  def create_blob_patches(self, blob):
    img = Image.open(blob.materialize())

    w = self.width
    h = self.height
    while w < img.size[0] and h < img.size[1]:
      for patch in self.create_sized_blob_patches(blob, img.size, w, h):
        yield patch
      w = int(w * self.scale)
      h = int(h * self.scale)

  def create_sized_blob_patches(self, blob, size, width, height):
    # How far each slide will go
    xstep = int(self.xoverlap * width)
    ystep = int(self.yoverlap * height)

    # The available room for sliding
    xdelta = size[0]-width
    ydelta = size[1]-height

    # How many slides, and how much unused "slide space"
    xsteps, extra_width = divmod(xdelta, xstep)
    ysteps, extra_height = divmod(ydelta, ystep)

    # Divy up the unused slide space, to lose evenly at edges
    left_indent = extra_width // 2
    top_indent = extra_height // 2

    for x in xrange(left_indent, xdelta, xstep):
      for y in xrange(top_indent, ydelta, ystep):
        patch = Patch.ifNew(blob=blob, x=x, y=y,
                            width=width, height=height, fliplr=False)
        if patch:
          yield patch
        if not self.fliplr:
          continue
        patch = Patch.ifNew(blob=blob, x=x, y=y,
                            width=width, height=height, fliplr=True)
        if patch:
          yield patch

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
      if self.params:
          self.instance.set_params(**self.params)

  @property
  def simple_class(self):
      return self.cls.split(".")[-1]
          
  def __repr__(self):
    return self.simple_class + "(" + str(self.params) + ")"

  @property
  def url(self):
    return "/featurespec/"+str(self.id)

  def create_dataset_features(self, ds):
    for blob in ds.blobs:
      for feature in self.create_blob_features(blob):
        yield feature

  def create_blob_features(self, blob):
    for patch in blob.patches:
      feat = self.create_patch_feature(patch)
      if feat:
          yield feat

  def create_patch_feature(self, patch):
    if Feature.query.filter_by(patch=patch, spec=self).count() > 0:
      return
    return Feature(patch=patch, spec=self,
                   vector=self.instance.extract(patch.image))

class Dataset(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  blobs = db.relationship("Blob", secondary=dataset_x_blob)
  patchspecs = db.relationship("PatchSpec",
                               secondary=dataset_x_patchspec,
                               backref="datasets")
  featurespecs = db.relationship("FeatureSpec",
                                 secondary=dataset_x_featurespec,
                                 backref="datasets")

  def migrate_to_s3(self):
    for blob in self.blobs:
      blob.migrate_to_s3()

  @property
  def url(self):
    return "/dataset/"+str(self.id)

  def __repr__(self):
    return model_debug(self)

  @property
  def images(self):
    return len(self.blobs)


class Keyword(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  name = db.Column(db.String)

  geoquery_id = db.Column(db.Integer, db.ForeignKey('geo_query.id'))
  geoquery = db.relationship('GeoQuery', backref = db.backref('keywords', lazy = 'dynamic'))

  @property
  def url(self):
    return "/keyword/"+self.name

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

  dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
  dataset = db.relationship('Dataset', backref = db.backref('classifiers', lazy = 'dynamic'))
  
  keyword_id = db.Column(db.Integer, db.ForeignKey('keyword.id'), nullable=False)
  keyword = db.relationship('Keyword', backref = db.backref('classifiers', lazy = 'dynamic'))

  estimator_id = db.Column(db.Integer, db.ForeignKey('estimator.id'), nullable=False)
  estimator = db.relationship('Estimator')

  def __init__(self, **kwargs):
      super(Classifier, self).__init__(**kwargs)
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
  def url(self):
    return "/classifier/"+str(self.id)

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
  def predict(self):
    '''Yield Predictions for all Patches in the Dataset, as of this Round.
    Creates an estimator for each FeatureSpec used by the Dataset,
    then trains each on the Features of that type.  Finally, uses each
    Estimator to make predictions against its Features.

    '''
    classifier = self.classifier

    estimators = {}
    for fs in classifier.dataset.featurespecs:
      samples, results = ([], [])
      for ex in classifier.examples_upto(self):
        feat = Feature.of(patch=ex.patch, spec=fs)
        samples.append(feat.vector)
        results.append(1 if ex.value else -1)

      estimators[fs.id] = classifier.estimator.instantiate()
      estimators[fs.id].fit(samples, results)

      for blob in classifier.dataset.blobs:
        for patch in blob.patches:
          for feature in patch.features:
            value = estimators[feature.spec.id].predict(feature.vector)
            yield Prediction(value=value, feature=feature, round=self)

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

class Patch(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  x = db.Column(db.Integer, nullable = False)
  y = db.Column(db.Integer, nullable = False)
  width = db.Column(db.Integer, nullable = False)
  height = db.Column(db.Integer, nullable = False)
  fliplr = db.Column(db.Boolean, nullable=False, default=False)
  rotation = db.Column(db.Float, nullable=False, default=0.0)

  blob_id = db.Column(db.Integer, db.ForeignKey('blob.id'), index = True)
  blob = db.relationship('Blob', backref = db.backref('patches', lazy = 'dynamic'))

  @property
  def size(self):
    assert self.width == self.height
    return self.width

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
        misc.imsave(complete, self.image)
    return complete

  @property
  def image(self):
    blob = self.blob
    with blob.open() as f:
      img = misc.imread(f)

      if self.fliplr:
        img = np.fliplr(img)
      if self.rotation != 0.0:
        img = rotate(img, self.rotation)

      crop = img[self.y:self.y+self.height, self.x:self.x+self.width]

      # Remove alpha channel if present
      if crop.shape[2] == 4:
        crop = crop[:,:,0:3]

      return crop

  @property
  def url(self):
    return "/patch/"+str(self.id);

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
    return str(self.spec.kind) + ' ' + str(self.vector)

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

class GeoQueryResult(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  value = db.Column(db.Float, nullable=False)
  gps_distance = db.Column(db.Float)
  
  geo_query_id = db.Column(db.Integer, db.ForeignKey('geo_query.id'), index = True, nullable=False)
  geo_query = db.relationship('GeoQuery', backref = db.backref('results', lazy = 'dynamic'))

  blob_id = db.Column(db.Integer, db.ForeignKey('blob.id'), nullable=False)
  blob = db.relationship('Blob', backref = db.backref('gq_results', lazy = 'dynamic'))

  def __repr__(self):
    return model_debug(self)

# Convenience routine simple repr implementation of models.
def model_debug(m):
  c = dict.copy(m.__dict__)
  del c['_sa_instance_state']
  return type(m).__name__+":"+str(c)

# This adapts (1-dimensional) numpy arrays to Postgres
# We should make it do n-dimensionals eventually.
from psycopg2.extensions import register_adapter, AsIs
def adapt_numpy(np):
    return AsIs(",".join([str(f) for f in np]))
register_adapter(np.ndarray, adapt_numpy)
