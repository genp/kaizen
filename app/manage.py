#!/usr/bin/env python
import config
from app import app, db
import os, traceback, subprocess, sys, time
from sqlalchemy.exc import OperationalError
import argparse
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from app.models import Blob, Dataset, Keyword, Classifier, Example, Prediction, HitResponse
#from lib.features import utils
import numpy as np
import signal
from contextlib import contextmanager
import psycopg2.extensions
from math import radians, cos, sin, asin, sqrt

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def start_process(cmd, args=None, job_name = 'job', use_grid = False):
    """
    Starts an independent process with a new group pid
    cmd is the string with full path to run script
    args is list of cmd arguments,
        each whitespace separeted item is a different list element
    returns integer: pid
    """

    # Start process
    if args:
        cmd = cmd+args
    print(cmd)

    # spawn grid process
    if use_grid:
        tmpFuncCall = " ".join(cmd)
        app.logger.info('Executing: %s' % tmpFuncCall)
        utils.launch_grid_job(job_name, tmpFuncCall, config.grid_logfileerr, config.grid_logfileout, job_length='vlong')
        return 'grid'
    else:
        out = config.subprocess_log_out+'_'+job_name+'.out'
        err = config.subprocess_log_err+'_'+job_name+'.err'
        os.system('touch '+out)
        os.system('touch '+err)
        process = subprocess.Popen(cmd,
                                   stdout=open(out, 'a'),
                                   stderr=open(err, 'a'),
                                   preexec_fn=os.setpgrp,
                                   close_fds = True
                                   )

        app.logger.info('Process launched: '+' '.join(cmd))
        app.logger.info('PID: '+str(process.pid))
        return process.pid

def get_hash_dir(root,name):
    dir_name = os.path.join(root, hashlib.md5(name.encode('utf-8')).hexdigest()[0:config.hash_length])
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def slugify(text, delim=u'-'):
    """
    Generates an ASCII-only slug
    """
    result = []
    for word in _punct_re.split(text.lower()):
        word = word.encode('translit/long')
        if word:
            result.append(word)
    return unicode(delim.join(result))

# calc features
def calculate_dataset_features(img_dir, save_dir, img_name = None, patch_details = None):


    # lists of cmd arguments
    grid = ['-g'] if config.grid_option else []
    features = ['-f']+config.feat_option if config.feat_option else []
    s3 = ['-s']+config.s3_option if config.s3_option else []
    email = ['-e']+config.email_option if config.email_option else []

    cmd = [os.path.join(config.clroot, 'bin/features/calc_dataset_feats.py'), img_dir, save_dir] + grid + features + s3 + email + ['-d']
    if img_name:
        single_file_option = ['-i']+[img_name]
        cmd = cmd + single_file_option
    if patch_details:
        patch_option = ['-p']+[str(val) for val in patch_details]
        cmd = cmd + patch_option

    job = Jobs(cmd=str(cmd), start_time=time.time(), isrunning=True, job_type='vlong')
    cmd = cmd + ['-j']+[str(job.id)]
    db.session.add(job)
    db.session.commit()

    # call script to calculate features
    app.logger.info('Calculating features for : '+img_dir+' '+save_dir)
    pid = start_process(cmd)
    app.logger.info('Adding patches to database from : '+save_dir)

    return pid

#Takes in a file extension, ex: .gif, .jpg, .png, and returns the MIME for that extension
#If the extension doesn't match one of the required extensions, returns "None"
def get_mime(extension):
    return config.mime_dictionary.get(extension, "None")


#create classifier
def create_classifier(keyword_id, dataset_id, classifier_name_prefix = None, seed_patch_ids = None):

    keyword = Keyword.query.get(keyword_id)
    dataset = Dataset.query.get(dataset_id)
    save_dir = get_hash_dir(config.CLASSIFIER_DIR, keyword.name)

    # init classifier, update predictions
    cmd = [os.path.join(config.clroot, 'lib/classifiers/active_learner.py')]
    args = ['-k', str(keyword_id), '-d', str(dataset_id),
            '--save', save_dir]
    if classifier_name_prefix:
        args += ['--name_prefix', classifier_name_prefix ]
    if seed_patch_ids:
        args += ['--seed_patch_ids', ' '.join(str(s) for s in seed_patch_ids)]

    app.logger.info('Creating classifier for keyword %d dataset %d: '%(keyword_id, dataset_id))
    cmd = cmd + args
    app.logger.info(str(cmd))
    job = Jobs(cmd=str(cmd), start_time=time.time(), isrunning=True, job_type='create_classifier')
    db.session.add(job)
    db.session.commit()
    # pid = start_process(cmd, args, job_name = 'create_classifier')
    # return pid

#update classifier, active queries
def update_classifier(classifier_id, hit_response_id, run_now = False):
    cmd = [os.path.join(config.clroot, 'lib/classifiers/active_learner.py')]
    args = ['-c', str(classifier_id), '-r', str(hit_response_id),
            '--update_classifier']
    app.logger.info('Updating classifier %d with hit_response %d '% (classifier_id, hit_response_id))
    app.logger.info(str(cmd+args))
    # This is for when ok_users are trying to bypass job queue
    if False: #TODO replace! #run_now:
        pid = start_process(cmd, args, 'update_c_'+str(classifier_id))
    else:
        cmd = cmd + args
        job = Jobs(cmd=str(cmd), start_time=time.time(), isrunning=True, job_type='update_hit')
        db.session.add(job)
        db.session.commit()

#update concensus hit response
def update_final_hit_response(classifier_id, hit_response_id):
    cmd = [os.path.join(config.clroot, 'lib/classifiers/active_learner.py')]
    args = ['-c', str(classifier_id), '-r', str(hit_response_id),
            '--update_hit_response']
    app.logger.info('Updating final hit_response with hit_response %d '% (hit_response_id))
    cmd = cmd + args
    job = Jobs(cmd=str(cmd), start_time=time.time(), isrunning=True, job_type='update_hit')
    db.session.add(job)
    db.session.commit()
    # pid = start_process(cmd, args, 'update_hit_'+str(hit_response_id))
    # return pid


#TODO: test classifier
def get_top_test_results(img_dir, save_dir, img_name = None, patch_details = None):

    top_patches = []
    # return top 200 predictions for this classifier, on given dataset
    return top_patches

#TODO: get catch trials
def get_catch_trails(img_dir, save_dir, img_name = None, patch_details = None):

    catch_trial_patches = []
    # return set of catch trials for this keyword
    return catch_trial_patches


# Returns Blob IDs for the estimated location nearest neighbors (in non-max suppressed, combined-value geo_query classifier space)
def get_geo_nearest_neighbors(geo_id):
    # result = db.engine.execute('select b.id, pre.classifier_id, max(value) from blob b, patch p, prediction pre where blob_id = b.id and pre.patch_id = p.id and pre.classifier_id in (select m.id from classifier m, keyword k, geo_query q where m.keyword_id = k.id and k.geoquery_id = q.id and q.id = '+str(geo_id)+') group by b.id, pre.classifier_id order by b.id').fetchall()

    # TODO: classifiers at different iterations
    classifiers = db.engine.execute('select m.id from classifier m, keyword k, geo_query q where m.keyword_id = k.id and k.geoquery_id = q.id and q.id = '+str(geo_id)).fetchall()
    result = []
    for c in classifiers:
        c_id = c[0]
        print(c_id)
        cur_c = Classifier.query.get(c_id)
        itr_id = cur_c.iteration_id
        result += db.engine.execute('select b.id, pre.classifier_id, max(value) from blob b, patch p, prediction pre where blob_id = b.id and pre.patch_id = p.id and pre.classifier_id = '+str(c_id)+' and pre.iteration_id = '+str(itr_id)+' group by b.id, pre.classifier_id order by b.id').fetchall()
        if len(result) == 0:
            print('using only patch queries')
            result += db.engine.execute('select b.id, pq.classifier_id, max(pred_value) from blob b, patch p, patch_query pq where blob_id = b.id and pq.patch_id = p.id and pq.classifier_id = '+str(c_id)+' and pq.iteration_id = '+str(itr_id)+' group by b.id, pq.classifier_id order by b.id').fetchall()
        else:
            print('using all predictions')
        print(len(result))

    result.sort(key=lambda x:(x[0], x[1]))

    nn_blobs = [[0,0]]
    cur_blob = 0
    cur_sum = 0
    for row in result:
        if row[0] == cur_blob:
            cur_sum += row[2]
        else:
            vals = map(lambda x: x[0], nn_blobs)

            if len(nn_blobs) < 20:
                nn_blobs = nn_blobs + [[cur_sum, cur_blob]]
            elif cur_sum > min(vals):
                nn_blobs[vals.index(min(vals))] = [cur_sum, cur_blob]

            cur_blob = row[0]
            cur_sum = row[2]


    nn_blobs.sort(key=lambda x:x[0], reverse=True)
    nn_blob_ids = [x[1] for x in nn_blobs]

    print(nn_blobs)

    # delete previous results
    old_results = GeoQueryResult.query.filter_by(geo_query_id = geo_id).all()
    for elem in old_results:
        db.session.delete(elem)

    keyword_id = db.engine.execute('select id from keyword where geoquery_id = '+str(geo_id)).fetchall()[0][0]
    query_blob_id = db.engine.execute('select p.blob_id from example e, patch p, blob b where p.blob_id = b.id and e.patch_id = p.id and e.classifier_id is null and e.keyword_id = '+str(keyword_id)).fetchall()[0][0]

    for b in nn_blobs:
        if b[1] == 0:
            continue
        blob1 = Blob.query.get(query_blob_id)
        blob2 = Blob.query.get(b[1])
        lon1 = blob1.longitude
        lon2 = blob2.longitude
        lat1 = blob1.latitude
        lat2 = blob2.latitude
        try:
            dist = haversine(lon1, lat1, lon2, lat2)
        except TypeError:
            ''' gps dist could not be calculated due to image gps coords missing '''
            dist = None
        result = GeoQueryResult(geo_query_id = geo_id, blob_id = b[1], value = b[0], gps_distance = dist)
        db.session.add(result)
    db.session.commit()

    return nn_blob_ids


#Returns Blob IDs for the ground-truth GPS nearest neighbors
def get_gps_nearest_neighbors(geo_id):
    keyword_id = db.engine.execute('select id from keyword where geoquery_id = '+str(geo_id)).fetchall()[0][0]
    query_blob_id = db.engine.execute('select p.blob_id from example e, patch p, blob b where p.blob_id = b.id and e.patch_id = p.id and e.classifier_id is null and e.keyword_id = '+str(keyword_id)).fetchall()[0][0]
    query_blob = Blob.query.get(query_blob_id)
    lon1, lat1 = (query_blob.longitude, query_blob.latitude)
    # todo: change '2' to the geolocation dataset
    all_blobs = db.engine.execute('select id, longitude, latitude from blob where id in (select distinct blob_id from patch where dataset_id = 2)').fetchall()

    nn_blobs = []
    for blob in all_blobs:
        dist = haversine(lon1, lat1, blob[1], blob[2])
        nn_blobs = nn_blobs + [[dist, blob[0]]]
    nn_blobs.sort(key=lambda x:x[0])
    nn_blobs = nn_blobs[:20]
    print(nn_blobs)
    for b in nn_blobs:
        if b[1] == 0:
            continue
        # value for ground truth geolocation neighbors is -100 to dinstingish them
        # from the estimated nearest neighbors that have positive values
        result = GeoQueryResult(geo_query_id = geo_id, blob_id = b[1], value = -100, gps_distance = b[0])
        db.session.add(result)
    db.session.commit()

    return nn_blobs


def get_lat_lon(img_name):
    image = Image.open(img_name)
    exif_data = get_exif_data(image)
    lat, lon = get_lat_lon_from_exif(exif_data)
    return lat, lon

# By Michael Dunn
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # 6367 km is the radius of the Earth
    earth_radius = 6367
    km = earth_radius * c
    return km

###################################
# Helper functions for get_lat_long
###################################


def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value

    return exif_data

def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None

def _convert_to_degress(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon_from_exif(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degress(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = _convert_to_degress(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

    return lat, lon

def add_blob(img_file):
    ext = os.path.splitext(img_file)[1]
    if ext == '.jpg':
        mime = 'image/jpeg'
        latitude, longitude = get_lat_lon(img_file)
    else:
        mime = 'image/'+ext.replace('.', '')
        latitude, longitude = (None, None)
    blob = Blob(location = img_file, ext = ext, mime = mime,
                latitude = latitude, longitude = longitude)
    db.session.add(blob)
    db.session.commit()

    return blob.id


def write_hit_file(search_prefix, hit_fname):
    classifiers = Classifier.query.filter(Classifier.name.contains(search_prefix)).all()
    ids = [c.id for c in classifiers]
    with open(hit_fname, 'w') as f:
        f.write('classifier_id\n')
        np.savetxt(f, ids, fmt='%d')
    print(str(len(ids))+' hits saved to '+hit_fname)

def proc_count(proc_name):
    cnt = int(os.popen('ps auxww | grep '+proc_name+' | grep -c -v grep').read())
    print('Count '+proc_name+' : '+str(cnt))
    return cnt

def wait_for(pred, interval = 60):
    '''
    waits for predicate to be true
    ex. use: wait_for(lambda : proc_count("stuff") < 5)
    '''
    while not pred():
        time.sleep(interval)
        sys.stdout.write('...')
        sys.stdout.flush()

def get_ids(stmt):
    return [c[0] for c in  db.engine.execute(stmt).fetchall()]

def classifier_update_daemon():
    wait_program = 'active_learner.py'
    try:
        try:
            with time_limit(60*60*10): # wait hours
                while 1:
                    job_id = db.engine.execute("select id from jobs where isrunning and not job_type = 'fashion_hit' order by start_time limit 1").fetchall()
                    while proc_count(wait_program) < config.proc_cap and job_id != []:
                        job_id = job_id[0][0]
                        job = Jobs.query.get(job_id)
                        cmd = eval(job.cmd)
                        if 'calc_dataset_feats' in cmd[0]:
                            job_name = 'calc_dataset_feats'
                        else:
                            job_name = utils.slugify(str(cmd[1:]))
                        pid = start_process(cmd, job_name=job_name)
                        db.engine.execute('delete from jobs where id = '+str(job.id))
                        job_id = db.engine.execute("select id from jobs where isrunning order by start_time limit 1").fetchall()

                    wait_for(lambda : proc_count(wait_program) < config.proc_cap, 5)
                    time.sleep(30)
                    sys.stdout.write('...')
                    sys.stdout.flush()

        except TimeoutException(msg):
             print("Cleaning up!")
             wait_for(lambda : proc_count(wait_program) == 0, 5)

             unfinished_classifiers = db.engine.execute('select distinct classifier_id from prediction').fetchall()
             # email this to myself
             message = 'unfinished classifiers '+str(unfinished_classifiers)
             utils.email_notify(message, '[Active Learner] daemon prediction clean up', config.admin_email)

             db.engine.execute('delete from prediction')
             conn = db.engine.connect()
             conn.connection.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
             conn.execute('VACUUM FULL PREDICTION;')
             conn.close()

             classifier_update_daemon()
    except:
        message = sys.exc_info()[0]
        print("Unexpected error:", message)
        message = OperationalError.message
        utils.email_notify(message, '[Active Learner] daemon crashed', config.admin_email)
        raise
if __name__ == "__main__":
    print('Launching update classifier daemon')
    classifier_update_daemon()
