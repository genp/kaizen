from app import app, db
from flask import render_template, redirect, url_for, jsonify, send_file
from app.forms import ActiveQueryForm, EvaluateForm
from app.models import User, Classifier, PatchQuery, PatchResponse, HitResponse, Estimator, Detection, Round, Dataset, Prediction, Blob, Patch, dataset_x_blob

import tasks

import time, itertools

@app.route('/evaluate/')
def evaluate_top():
    eval_form = EvaluateForm()
    return render_template('evaluate_top.html',
                            title="Evaluate",
                            eval_form=eval_form)

@app.route('/evaluate-ds/', methods = ['POST'])
def evaluate_ds():
    eval_form = EvaluateForm()
    if eval_form.validate_on_submit:
        round = eval_form.round.data
        dataset = eval_form.dataset.data

        max_patch2ds = db.engine.execute('SELECT MAX(id) as max_id FROM patch WHERE blob_id in \
                                    (SELECT blob_id FROM dataset_x_blob \
                                    WHERE dataset_id = %d);' % dataset.id).fetchone()
        present_in_val = db.engine.execute('SELECT id as id from val_prediction WHERE patch_id=%d;' % max_patch2ds['max_id']).fetchone()

        if present_in_val:
            return redirect(url_for('classifier_evaluate', classifier_id=round.classifier_id,
                                    round_id=round.id, dataset_id=dataset.id))
        else: #generate predicitions, url to somewhere else
            tasks.predict_ds.delay(round.id, dataset.id)
    return redirect(url_for('evaluate_top'))#rendering conditions
