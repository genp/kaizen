from app import app, db, manage
from flask import render_template, redirect, request, url_for, g, jsonify, send_file
from flask_login import login_user, logout_user, login_required
from app.forms import *
from app.models import User, Blob, Patch
import config
import os, tempfile
from app import dataset_views, keyword_views, classifier_views, evaluate_views

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.route('/appname', methods=['GET'])
def get_appname():
    return config.APPNAME

@app.route('/', methods=['GET','POST'])
def top():
    return render_template('top.html')

@app.route('/user/<username>')
#@login_required
def user(username):
    user = User.find(username)
    if user == None:
        return redirect(url_for('top'))
    return render_template('user.html', user = user)

@app.route('/blob/<int:id>')
def blob(id):
    blob = Blob.query.get_or_404(id)
    return redirect(blob.url)

@app.route('/blob/<int:id>/debug')
def blob_debug(id):
    blob = Blob.query.get_or_404(id)
    return render_template('blob.html', blob=blob)


@app.route('/patch/<int:id>')
def patch(id):
    patch = Patch.query.get_or_404(id)
    return send_file(patch.materialize());

@app.route('/patch/<int:id>/debug')
def patch_debug(id):
    patch = Patch.query.get_or_404(id)
    return render_template('patch.html', patch=patch)


@app.route('/file-upload/', methods = ['POST'])
def file_upload():
    form = BlobForm()

    if form.validate_on_submit():
        print("Uploaded file: "+form.file.data.filename)

        upload = form.file.data
        _, ext = os.path.splitext(upload.filename)
        with tempfile.NamedTemporaryFile(dir = config.BLOB_DIR,
                                         prefix='image', suffix = ext,
                                         delete = False) as tmp:
            filename = os.path.basename(tmp.name)
            upload.save(tmp)
        blob = Blob(tmp.name)
        db.session.add(blob)
        db.session.commit();
        return jsonify(results=blob.id)
    else:
        return jsonify(results=0, errors=form.file.errors)
