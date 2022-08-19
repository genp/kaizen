'''

Sets up global variables for the app.

'''
import os,sys,socket
from dotenv import load_dotenv
load_dotenv()

"""

Server resource paths and hosting options

"""

approot = os.getenv('APPROOT')
if not approot:
    approot = os.path.dirname(os.path.abspath(__file__))
sys.path.append(approot)


DEVELOPMENT = False

HOST = '' #'localhost' # AWS example: 'ec2-54-175-135-17.compute-1.amazonaws.com' #socket.getfqdn()
if 'local' in HOST:
    HOST = 'localhost'
PORT = 8888
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

"""

AWS Options

"""
# If this is set to true, user running the app must have AWS CLI config set up
# for an IAM user that has permission to write to the s3 bucket named
# APPNAME-blobs
# All uploaded images and videos will be saved to that bucket
USE_AWS_S3 = False


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
# the active query strategy
active_query_strategy='most_confident'

'''
Flask App
'''

APPNAME = 'kaizen'
DEBUG = True
CSRF_ENABLED = True
SECRET_KEY = 'robots are people too'

basedir = os.path.abspath(os.path.dirname(__file__))

BLOB_DIR = os.path.join(basedir, 'app', 'static', 'blobs')
DATASET_DIR = os.path.join(basedir, 'app', 'static', 'datasets')
CACHE_DIR = os.path.join(basedir, 'app', 'static', 'cache')
LOG_DIR = os.path.join(approot, 'app', 'static', 'logs')
LOG_FILE = os.path.join(LOG_DIR, APPNAME+'.log')

#Remote DB ex: SQLALCHEMY_DATABASE_URI = 'postgresql://'+user+'@localhost/'+APPNAME

POSTGRES = {
        'user':  os.getenv("USER"),
        'pw': os.getenv("USER"),
        'db': APPNAME+'-local',
        'host': 'localhost',
        'port': '5432',
    }
SQLALCHEMY_DATABASE_URI = 'postgresql://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES

SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
SQLALCHEMY_TRACK_MODIFICATIONS = False

BROKER_URL="sqla+"+SQLALCHEMY_DATABASE_URI
CELERY_RESULT_BACKEND="db+"+SQLALCHEMY_DATABASE_URI


# mail server settings
MAIL_SERVER = 'localhost'
MAIL_PORT = 25
MAIL_USERNAME = None
MAIL_PASSWORD = None

# administrator list
ADMINS = ['name@email.com']

USER_ENABLE_CONFIRM_EMAIL = False
USER_ENABLE_EMAIL = False

# Used to retrieve meta-data from ec2 machines
def ec2_metadata(tag):
    md_cmd = 'curl -s --connect-timeout 1 http://<ip-address>/latest/meta-data/'
    return os.popen(md_cmd+tag).read();

# set logging level to 2 to suppress caffe output
os.environ['GLOG_minloglevel'] = '2'
#USE_GPU = False
#instance_type = ec2_metadata('instance-type')
#EC2 = instance_type != ''
#if instance_type.startswith("g"):
USE_GPU = True
GPU_DEVICE_IDS = [0]

EC2 = False
for dir in (BLOB_DIR, DATASET_DIR, CACHE_DIR, LOG_DIR):
    sub = os.path.basename(dir)
    if not os.path.exists(dir):
        if EC2:
            os.system('sudo mkdir -p /mnt/$USER')
            os.system('sudo chown $USER /mnt/$USER')
            os.system('mkdir -p /mnt/$USER-space/'+sub)
            os.system('ln -sf /mnt/$USER-space/'+sub+' '+dir)
        else:
            os.mkdir(dir)
