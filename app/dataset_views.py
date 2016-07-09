from app import app, db
from flask import render_template, g, redirect, jsonify
from forms import *
from models import Dataset, Blob, PatchSpec, FeatureSpec, Classifier
import tempfile
import config
import os
import json
import zipfile, tarfile, csv
import random
import tasks

def as_url(str):
    if str.startswith("http://"):
        return str
    if str.startswith("https://"):
        return str
    if str.startswith("ftp://"):
        return str
    if str.startswith("www."):
        return "http://"+str
    return None

@app.route('/dataset-upload/', methods = ['POST'])
def dataset_upload():
    form = DatasetForm()
    if form.validate_on_submit():
        upload = form.file.data
        name, ext = os.path.splitext(upload.filename)

        acceptable = ['.jpg', '.jpeg', '.png']
        label_acceptable = ['.csv']

        def unarchive_blob(fname, dset, tmpd, archive):
            _, ext = os.path.splitext(fname)
            if ext in acceptable and not fname.startswith("__MACOSX"):
                archive.extract(item, tmpd)
                blob = Blob(str(tmpd) + "/" + fname)
                dset.blobs.append(blob)
            return

        def list_blob(url):
            _, ext = os.path.splitext(url)
            if ext in acceptable:
                blob = Blob(url)
                dset.blobs.append(blob)
            return

        def keyword_dataset(fname,dset, tmpd, archive):
            kw, ext = os.path.splitext(fname)
            if ext in label_acceptable:
                # TODO: unarchive the csv file and store to tmpd
                kw_fname = str(tmpd) + "/" + fname

                k = Keyword(name=kw[1:],defn_file=kw_fname,dataset=dset)
                dset.blobs.append(blob)
            return

        dset = None
        if ext == ".zip":
            with zipfile.ZipFile(upload, 'r') as myzip:
                tmpd = tempfile.mkdtemp(dir = config.DATASET_DIR,
                                        prefix = "dataset")
                dset = Dataset(name = name)
                db.session.add(dset)

                for item in myzip.infolist():
                    _, ext = os.path.splitext(item.filename)
                    if ext in acceptable and not item.filename.startswith("__MACOSX"):
                        myzip.extract(item, tmpd)
                        blob = Blob(str(tmpd) + "/" + item.filename)
                        dset.blobs.append(blob)
        elif ext == ".gz" or ext == ".bz2" or ext == ".tar":
            if ext != ".tar":
                name, ext = os.path.splitext(name)

            if ext == ".tar":
                with tarfile.open(fileobj=upload) as mytar:
                    tmpd = tempfile.mkdtemp(dir = config.DATASET_DIR,
                                            prefix = "dataset")
                    dset = Dataset(name = name)
                    db.session.add(dset)

                    for item in mytar:
                        if item.isreg():
                            _, ext = os.path.splitext(item.name)
                            if ext in acceptable and not item.name.startswith("__MACOSX"):
                                mytar.extract(item, tmpd)
                                blob = Blob(str(tmpd) + "/" + item.name)
                                dset.blobs.append(blob)
        # TODO move this up so that the zip and tar files could also contain specs in a csv or txt file
        elif ext == ".txt":
            dset = Dataset(name = name)
            db.session.add(dset)
            for url in upload:
                url = url[:-1]
                _, ext = os.path.splitext(url)
                if ext in acceptable:
                    blob = Blob(url)
                    dset.blobs.append(blob)
        elif ext == ".csv":
            dset = Dataset(name = name)
            db.session.add(dset)
            for row in csv.reader(upload):
                for entry in row:
                    url = as_url(entry)
                    if url:
                        _, ext = os.path.splitext(url)
                        if ext in acceptable:
                            blob = Blob(url)
                            dset.blobs.append(blob)


        if dset != None:
            if form.patchspec.data:
                dset.patchspecs.append(form.patchspec.data)
            if form.featurespec.data:
                dset.featurespecs.append(form.featurespec.data)
            db.session.commit()
            tasks.dataset.delay(dset.id)
            return jsonify(name=dset.name, id=dset.id, url = dset.url)
    else:
        print form.errors
        return jsonify(errors=form.file.errors)



@app.route('/dataset/attach', methods = ['POST'])
def dataset_attach():
    form = DatasetAddSpecsForm()
    dset = form.dataset.data
    if form.patchspec.data:
        dset.patchspecs.append(form.patchspec.data)
    if form.featurespec.data:
        dset.featurespecs.append(form.featurespec.data)
    db.session.commit()
    tasks.dataset.delay(dset.id)
    return redirect(dset.url)

@app.route('/dataset/', methods = ['GET'])
def dataset_top():
    datasets = Dataset.query.all()
    return render_template('dataset_top.html',
                           datasets = datasets)


@app.route('/dataset/new', methods = ['GET', 'POST'])
def dataset_new():
    dataset_form = DatasetForm()
    return render_template('dataset_new.html',
                           title = 'New Dataset',
                           dataset_form = dataset_form)

@app.route('/dataset/<int:id>', methods = ['GET'])
def dataset(id, psform=None, fsform=None):
    dataset = Dataset.query.get_or_404(id)
    classifiers = Classifier.query.filter_by(dataset_id = id)

    num_ex = 20
    blobs = dataset.blobs[:num_ex]
    if not psform:
        psform = PatchSpecForm(dataset=dataset)
    if not fsform:
        fsform = FeatureSpecForm(dataset=dataset)

    addpsform = DatasetAddSpecsForm(dataset=dataset)
    addfsform = DatasetAddSpecsForm(dataset=dataset)

    return render_template('dataset.html',
                           title = dataset.name,
                           dataset = dataset,
                           blobs = blobs,
                           classifiers = classifiers,
                           addpsform=addpsform, psform = psform,
                           addfsform=addfsform, fsform = fsform)

@app.route('/patchspec/', methods = ['GET'])
def patchspec_top(psform=None):
    patchspecs = PatchSpec.query.all()
    if not psform:
        psform = PatchSpecForm()
    return render_template('patchspec_top.html',
                           patchspecs = patchspecs,
                           psform = psform)

@app.route('/patchspec/new', methods = ['GET', 'POST'])
def patchspec_new():
    psform = PatchSpecForm()
    next = "/patchspec/";
    if psform.validate_on_submit():
        pspec = PatchSpec(name=psform.name.data,
                          width=psform.width.data,
                          height=psform.height.data,
                          scale=psform.scale.data,
                          xoverlap=psform.xoverlap.data,
                          yoverlap=psform.yoverlap.data,
                          fliplr=psform.flip.data)
        db.session.add(pspec)
        if psform.dataset.data:
            psform.dataset.data.patchspecs.append(pspec)
            next = psform.dataset.data.url
        db.session.commit()
        tasks.if_dataset(psform.dataset.data)
        return redirect(next)
    if psform.dataset.data:
        return dataset(psform.dataset.data.id, psform=psform)
    else:
        return patchspec_top(psform=psform)


@app.route('/featurespec/', methods = ['GET'])
def featurespec_top():
    featurespecs = FeatureSpec.query.all()
    fsform = FeatureSpecForm()
    return render_template('featurespec_top.html',
                           featurespecs = featurespecs,
                           fsform = fsform)

@app.route('/featurespec/new', methods = ['GET', 'POST'])
def featurespec_new():
    fsform = FeatureSpecForm()
    next = "/featurespec/";
    if fsform.validate_on_submit():
        fspec = FeatureSpec(name=fsform.name.data,
                            cls=fsform.cls.data,
                            params=fsform.params.data)
        db.session.add(fspec)
        if fsform.dataset.data:
            fsform.dataset.data.featurespecs.append(fspec)
            next = fsform.dataset.data.url
        db.session.commit()
        tasks.if_dataset(fsform.dataset.data)
        return redirect(next)
    if fsform.dataset.data:
        return dataset(fsform.dataset.data.id, fsform=fsform)
    else:
        return featurespec_top(fsform=fsform)
