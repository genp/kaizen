from app import app, db, lm, manage
from flask import make_response, render_template, flash, redirect, request, url_for, g, jsonify
from forms import *
from models import User, Blob, Dataset, Keyword, Classifier, Patch, Example, Prediction, GeoQuery, PatchQuery, PatchResponse, HitResponse, GeoQueryResult
from  sqlalchemy.sql.expression import func
import tempfile
import config
import os
import json
import zipfile
import random, time
import numpy as np

@app.route('/geo/', methods = ['GET'])
def geo_top():
    geoquery = GeoQuery.query.all()
    return render_template('geo_top.html', title="Geolocation", geo = geoquery)

@app.route('/geo/new/', methods = ['GET'])
def geo_new():
    form = SeedForm()
    blob_form = BlobForm()
    return render_template('geo_new.html', title='Geolocation Library', form=form, blob_form=blob_form)

@app.route('/geo/add/', methods = ['POST'])
def geo_add():
    form = SeedForm() 
    if form.validate_on_submit():                
        geo_query = GeoQuery(name=form.keyword.data)
        db.session.add(geo_query)
        seeds = json.loads(form.seeds.data)

        for blob_id in seeds.keys():
            count = 0
            for patch in seeds[blob_id]:
                
                keyword = Keyword(name=form.keyword.data+str(count), geoquery = geo_query)
                count += 1
                db.session.add(keyword)

                x = max(0, patch[0])
                y = max(0, patch[1])
                size = patch[2]
                print '%s (%d, %d, %d)' % (blob_id, x, y, size)
                blob = Blob.query.get(blob_id)
                patch = Patch(blob = blob, x = int(x), y = int(y), size = int(size))
                db.session.add(patch)
                seed = Example(value = True, patch = patch, keyword = keyword)
                db.session.add(seed)
                db.session.commit()
                
                # calculate patch feature, save feature file with just that patch
                manage.calculate_dataset_features(config.BLOB_DIR, config.FEATURES_DIR, os.path.basename(blob.location), [x, y, size, patch.id])

                # initialize classifier, change to select geo_query dataset                
                manage.create_classifier(keyword_id = keyword.id, dataset_id = 2, num_seeds = 1)
                

    else:
        print 'did not validate'
        print form.keyword.errors
        print form.seeds.errors

    return redirect(url_for('geo_top'))

# TODO: super super slow
@app.route('/geo/<id>/', methods = ['GET'])
def geo(id):
    print 'geoquery: '+id
    id = int(id)
    geo_query = GeoQuery.query.get(id).name
    classifier_ids = db.engine.execute('select m.id from classifier m, keyword k, geo_query q where m.keyword_id = k.id and k.geoquery_id = q.id and q.id = '+str(id)).fetchall()
    print geo_query
    print classifier_ids
    patches = {}
    for mid in classifier_ids:
        # get the seed for this classifier (classifier = mid, keyword is something (the keyword for classifier mid))
        print mid
        classifier = Classifier.query.get(mid[0])
        keyword = Keyword.query.get(classifier.keyword_id)
        patch_id = db.engine.execute('select ex.patch_id from example ex where ex.keyword_id = '+str(keyword.id)+' and ex.classifier_id is null').fetchall()
        # patch_id = db.engine.execute('select ex.patch_id from example ex where ex.classifier_id = '+str(mid[0])+' and ex.keyword_id is not null').fetchall();
        p = Patch.query.get(patch_id[0][0])
        try:
            patches[p.blob.id].append([p.x, p.y, p.size, mid[0]])
        except:
            patches[p.blob.id]=[[p.x, p.y, p.size, mid[0]]]

    # Nearest Neighbor blob ids
    neighbors = GeoQueryResult.query.filter_by(geo_query_id = id)
    neighbors = [n.blob_id for n in neighbors]
    print neighbors
    print 'rendering geo '+str(id)
    # for each seed, the current active query (or training pending notice)
    print patches
    return render_template('geo.html', keyword=geo_query, title=geo_query, 
                           patches=patches, keyword_id=id, queryImgName=patches.keys()[0],
                           neighbors = neighbors)
