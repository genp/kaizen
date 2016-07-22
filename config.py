'''

Sets up global variables for Kaizen.

'''
import os,sys,socket
kairoot = os.getenv('KAIROOT')
if not kairoot:
    kairoot = os.path.dirname(os.path.abspath(__file__))
sys.path.append(kairoot)

cafferoot = os.getenv('CAFFEROOT')
if not cafferoot:
    cafferoot = '~/caffe'
sys.path.append(cafferoot)
DEVELOPMENT = False

HOST = socket.getfqdn()
if 'local' in HOST:
    HOST = 'localhost'
PORT = 8080
URL_PREFIX = 'http://'+HOST
if PORT != 80:
    URL_PREFIX = URL_PREFIX+':'+str(PORT)
URL_PREFIX = URL_PREFIX+'/'


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

# for dir in (BLOB_DIR, DATASET_DIR, CACHE_DIR, LOG_DIR):
#     if not os.path.exists(dir):
#         os.mkdir(dir)

if not os.path.exists(BLOB_DIR):
    os.system('sudo mkdir /mnt/$USER')
    os.system('sudo chown $USER /mnt/$USER')
    os.system('mkdir /mnt/$USER/kaizen-space')
    os.system('mkdir /mnt/$USER/kaizen-space/datasets')
    os.system('mkdir /mnt/$USER/kaizen-space/cache')
    os.system('mkdir /mnt/$USER/kaizen-space/blobs')
    os.system('ln -sf /mnt/$USER/kaizen-space/datasets')
    os.system('ln -sf /mnt/$USER/kaizen-space/cache')
    os.system('ln -sf /mnt/$USER/kaizen-space/blobs')


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

# Used to retrieve meta-data from ec2 machines
def ec2_metadata(tag):
    md_cmd = 'curl -s --connect-timeout 1 http://169.254.169.254/latest/meta-data/'
    return os.popen(md_cmd+tag).read();

# set logging level to 2 to suppress caffe output
os.environ['GLOG_minloglevel'] = '2'
USE_GPU = False
instance_type = ec2_metadata('instance-type')
if instance_type.startswith("g"):
    print "Using GPU"
    USE_GPU = True
    GPU_DEVICE_ID = 0
