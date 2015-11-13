from app import app, db
from flask import render_template, redirect, request, url_for, g, jsonify
from forms import ClassifierForm, SeedForm
from models import User, Blob, Dataset, Keyword, Classifier, Patch, Example
import json

@app.route('/keyword/', methods = ['GET'])
def keyword_top():
    form = SeedForm()
    keywords = Keyword.query.order_by(Keyword.id.desc()).filter(Keyword.seeds.any()).limit(50)
    return render_template('keyword_top.html', 
                           title = 'Keyword Library',
                           keywords = keywords,
                           classifiers = [], #classifiers,
                           form = form)

@app.route('/keyword/new', methods = ['GET'])
def keyword_new():
    form = SeedForm()
    blob_form = BlobForm()
    return render_template('keyword_new.html',
                           title='New Keyword', form=form, blob_form=blob_form)

@app.route('/keyword/add', methods = ['POST'])
def keyword_add():
    form = SeedForm() 
    if form.validate_on_submit():        
        keyword = Keyword.query.filter_by(name=form.keyword.data).first()
        keyword = keyword if not keyword==None else Keyword(name=form.keyword.data)

        db.session.add(keyword)

        seeds = json.loads(form.seeds.data)

        for blob_id in seeds.keys():
            blob = Blob.query.get_or_404(blob_id)
            for patch in seeds[blob_id]:                
                (x, y, size) = patch
                print '%s (%d, %d, %d)' % (blob_id, x, y, size)
                patch = Patch(blob = blob, x = int(x), y = int(y),
                              width = int(size), height = int(size))
                db.session.add(patch)
                seed = Example(value = True, patch = patch, keyword = keyword)
                db.session.add(seed)
                # We don't create features yet, because we don't know
                # what datasets, and thus what patches and features
                # we'll be running against.
                # feat = fs.create_patch_feature(patch)
                # db.session.add(feat)
        db.session.commit()

        return redirect(url_for('keyword', id=keyword.id))
    else:
        print 'did not validate'
        print form.keyword.errors
        print form.seeds.errors
        return redirect(url_for('keyword_new'))
    return redirect(url_for('keyword_top'))

@app.route('/keyword/<int:id>', methods = ['GET'])
def keyword(id):
    keyword = Keyword.query.get_or_404(id)
    examples = keyword.seeds

    # Form for creating new classifier for this keyword
    form = ClassifierForm()
    form.keyword.data = keyword

    return render_template('keyword.html', title=keyword.name,
                           keyword=keyword, form=form)

