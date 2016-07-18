'''

Sets up global variables for Kaizen.

'''
import os,sys
kairoot = os.getenv('KAIROOT')
if not kairoot:
    kairoot = os.path.dirname(os.path.abspath(__file__))
sys.path.append(kairoot)

cafferoot = os.getenv('CAFFEROOT')
if not cafferoot:
    cafferoot = '~/caffe'
sys.path.append(cafferoot)
DEVELOPMENT = False


user = os.environ['USER']

mime_dictionary = {
  ".jpg" : "image/jpeg",
  ".jpeg" : "image/jpeg",
  ".gif" : "image/gif",
  ".png" : "image/png"
}


# bow vocabularies
if False:                       # Bring this back if needed, from crowd_learner
    import numpy as np
    from sklearn import cluster
    from sklearn.neighbors import NearestNeighbors
    color_clusterer = cluster.MiniBatchKMeans(n_clusters=vocab_size)
    color_clusterer.cluster_centers_ = np.load(cluster_file)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(color_clusterer.cluster_centers_)

# empty patch classifier
if False:                       # Bring this back if needed, from crowd_learner
    import sys

    sys.path.append(os.path.join(kairoot, 'bin/empty_patch/'))
    # import ep_classifier

    # epc = ep_classifier.EmptyPatchClassifier(os.path.join(kairoot, 'bin/empty_patch'))
    # epc.load()

"""

Classifier params

"""

classifier_type = 'linear svm'

lin_svm = dict(
    C = 1.0,
    dual = True,
    verbose = True
)

threshold = -1.0

# How many of the predictions to ask about in one round
query_num = 200

'''
Flask App
'''

APPNAME = 'kaizen'
DEBUG = True
CSRF_ENABLED = True
SECRET_KEY = 'crowds are people too'

basedir = os.path.abspath(os.path.dirname(__file__))

BLOB_DIR = os.path.join(basedir, 'app', 'static', 'blobs')
DATASET_DIR = os.path.join(basedir, 'app', 'static', 'datasets')
CACHE_DIR = os.path.join(basedir, 'app', 'static', 'cache')
LOG_DIR = os.path.join(kairoot, 'app', 'static', 'logs')
LOG_FILE = os.path.join(LOG_DIR, APPNAME+'.log')

for dir in (BLOB_DIR, DATASET_DIR, CACHE_DIR, LOG_DIR):
    if not os.path.exists(dir):
        os.mkdir(dir)

SQLALCHEMY_DATABASE_URI = 'postgresql://'+user+'@localhost/'+APPNAME
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

BROKER_URL="sqla+"+SQLALCHEMY_DATABASE_URI
CELERY_RESULT_BACKEND="db+"+SQLALCHEMY_DATABASE_URI

# mail server settings
MAIL_SERVER = 'localhost'
MAIL_PORT = 25
MAIL_USERNAME = None
MAIL_PASSWORD = None

# administrator list
ADMINS = ['gen@cs.brown.edu']

USER_ENABLE_CONFIRM_EMAIL = False
USER_ENABLE_EMAIL = False

# set logging level to 2 to suppress caffe output
os.environ['GLOG_minloglevel'] = '2'
