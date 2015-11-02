'''

Sets up global variables for Kaizen.

'''
import os
clroot = os.getenv('CLROOT')
if not clroot:
    clroot = os.path.dirname(os.path.abspath(__file__))
DEVELOPMENT = False


user = os.environ['USER']

"""
Parameters for feature calculation.
"""

mime_dictionary = {
  ".jpg" : "image/jpeg",
  ".jpeg" : "image/jpeg",
  ".gif" : "image/gif",
  ".png" : "image/png"
}


# this is to make smaller directories when saving feature files
split_save_path = True

patch_size = 75.0 #px square
bin_split = 10.0
sbin = int(5) # this has to do with Pedro's HoG params, patch_size/bin_split
padx = 0
pady = 0

# NOTE: this is kind of a big step size...
step_percent = 0.25 # 0.5 # step size as a percent of the patch window
hog_patch_size = patch_size/sbin - 2


# color params
bow_num_train = 500
tiny_img_size = 100.0
vocab_size = 256

#cluster_file = os.path.join(clroot, 'data/features/color_bow_cluster_centers_'+str(vocab_size)+'.npy')

#cluster_file = 'cub_data/color_bow_cluster_centers_'+str(vocab_size)+'.npy'

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

    sys.path.append(os.path.join(clroot, 'bin/empty_patch/'))
    # import ep_classifier

    # epc = ep_classifier.EmptyPatchClassifier(os.path.join(clroot, 'bin/empty_patch'))
    # epc.load()

# img feature params
scales = [75.0, 100.0, 150.0, 200.0] #[50.0, 75.0, 100.0, 150.0, 200.0, 250.0]
# max and min for sliding patch size as a
# percentage of the input image's minimum dimension
patch_percent = [1.0, 0.1]

# compression factor for saving feature files
compression = 6

# DecafNet params
decaf_dir = '/home/gen/decaf-release/'
decaf_net = os.path.join(decaf_dir, 'imagenet.decafnet.epoch90')
decaf_meta = os.path.join(decaf_dir,'imagenet.decafnet.meta')
blob_name = 'fc6_cudanet_out'
# for normalized cnn feat
cnn_dim = 200
# HoG Pydro params


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

# values for AP calculation
_overlap_ratio_thresh = 0.3
start = 10
step = 10
_N = [start+i*step for i in range(10)]
_thresh = 0.0

# catch trials - even split positives and negatives
num_catches = 5

# How many of the predictions to ask about in one round
query_num = 200

grid_logfileout = '%sapp/static/tmp/grid.out' % (clroot)#
grid_logfileerr = '%sapp/static/tmp/grid.err' % (clroot)#'/dev/null'#

# can run #<proc_cap> classifier updates in parallel
grid_cap = 350
proc_cap = 10
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
PATCH_DIR = os.path.join(basedir, 'app', 'static', 'patches')
FEATURES_DIR = os.path.join(basedir, 'app', 'static', 'features')
CLASSIFIER_DIR = os.path.join(basedir, 'app', 'static', 'classifiers')
SYNTH_DIR = os.path.join(basedir, 'app', 'static', 'synthetics')

for dir in (BLOB_DIR, DATASET_DIR, PATCH_DIR, FEATURES_DIR, CLASSIFIER_DIR, SYNTH_DIR):
    if not os.path.exists(dir):
        os.mkdir(dir)

SQLALCHEMY_DATABASE_URI = 'postgresql://'+user+'@localhost/kaizen'
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

# default image features for the app
feat_option = ['cnn']
grid_option = False
s3_option = None
email_option = ['gen@cs.brown.edu']

log_path = os.path.join(clroot, 'app', 'static', 'tmp')
log_file = os.path.join(log_path, APPNAME+'.log')

subprocess_log_out = '/home/gen/crowd_learner/app/static/tmp/subprocess'
subprocess_log_err = '/home/gen/crowd_learner/app/static/tmp/subprocess'

# for creating hashed directory names
hash_length = 4

app_classifier_type = 'lin_svm'
app_feature_type = 'cnn'
server_use_grid = False

admin_email = 'gen@cs.brown.edu'

# MTurk HIT params
bbox_overlap = 0.3#0.1

# HITs are repeated for this many workers
hit_repeat = 3 #9
mturk_save_path = '/home/gen/crowd_learner/mturk/active_query_task/classifiers/'
mturk_form_path = '/home/gen/crowd_learner/mturk/active_query_task/active_query'

# other training variables
scene_dataset_id = 2
CUB_train_dataset_id = 20
hierarchy_fname = os.path.join(clroot, 'data/similarity/hierarchy.jbl')
patch_loc_file = '/data/hays_lab/CUB_200_2011/head_patch_locs.pkl'
im_file = '/data/hays_lab/CUB_200_2011/images.txt'
train_test_split_file = '/data/hays_lab/CUB_200_2011/train_test_split.txt'
max_iteration_number = 5
testset_id = 30
