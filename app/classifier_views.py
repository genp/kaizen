from app import app, db
from flask import render_template, redirect, url_for, jsonify

from forms import ActiveQueryForm, ClassifierForm
from models import User, Classifier, PatchQuery, PatchResponse, HitResponse, Estimator

import tasks

import time, itertools

@app.route('/classifier/', methods = ['GET', 'POST'])
def classifier_top():
    classifiers = Classifier.query.order_by(Classifier.id.desc()).limit(50)
    return render_template('classifier_top.html',
                           title='Classifier Library',
                           classifiers=classifiers)
    
@app.route('/classifier/<int:id>', methods = ['GET'])
def classifier(id):
    classifier = Classifier.query.get_or_404(id)

    round = classifier.rounds[-1]
    hits = make_hit(classifier.rounds[0].examples, round.queries)

    form = ActiveQueryForm()
    form.classifier.data = classifier
    form.round.data = round

    return render_template('classifier.html', classifier=classifier,
                           title='%s %d' % (classifier.keyword.name,
                                            classifier.rounds.count()),
                           form=form, hits = hits)

@app.route('/classifier/<int:id>/<int:i>/', methods = ['GET'])
def classifier_round(id, i):
    classifier = Classifier.query.get_or_404(id)

    round = classifier.rounds[i]
    hits = make_hit(classifier.keyword.seeds, round.queries)

    form = ActiveQueryForm()
    form.classifier.data = classifier
    form.round.data = round

    return render_template('classifier_round.html', classifier=classifier,
                           title='%s %d' % (classifier.keyword.name, i),
                           form=form, round = round, hits = hits)

@app.route('/classifier/update/<int:id>', methods = ['POST'])
def classifier_update(id):
    form = ActiveQueryForm();

    if form.validate_on_submit():
        print 'form.user.data '+form.user.data
        user = current_user

        if user.is_anonymous and form.user.data:
            user = User.find(form.user.data)
            if user:
                assert user.password == None
            else:
                user = User(username=form.user.data, password=None)
                db.session.add(user)
        
                user.location = form.location.data
        user.nationality = form.nationality.data
        print 'update classifier %s from %s.....' % (id, user)
        
        hit_resp = HitResponse(time = form.time.data,
                               confidence = form.confidence.data,
                               user = None if user.is_anonymous else user)
        db.session.add(hit_resp)
        db.session.commit()
        print str(time.time()) + ": " + str(hit_resp)
        
        pos_response = [(p, True) for p in form.pos_patches.data]
        neg_response = [(p, False) for p in form.neg_patches.data]

        for patch, v in itertools.chain(pos_response, neg_response):
            pq = PatchQuery.query.filter_by(patch=patch, round=form.round.data).one()
            pr = PatchResponse(value=v, hitresponse=hit_resp, patchquery = pq)
            db.session.add(pr)

        print str(time.time()) + ": done"

        db.session.commit()
        print str(time.time()) + ": committed"
        tasks.advance_classifier.delay(form.classifier.data.id)
        
    else:               
        for field in form:
            print field.name
            print field.errors
            print field.data
    return jsonify(results='success')

@app.route('/classifier/new', methods = ['POST'])
def classifier_new():
    form = ClassifierForm()    

    if form.validate_on_submit():
        c = Classifier(dataset = form.dataset.data,
                       keyword = form.keyword.data,
                       estimator = form.estimator.data)
        db.session.add(c)
        db.session.commit()
        tasks.if_classifier(c)
        return redirect(c.url)
    else:
        print 'did not validate'
        print form.dataset.errors
        return redirect(url_for('classifier_top'))

@app.route('/classify/', methods = ['GET', 'POST'])
def classify():
    pass
   
def make_hit(examples, patch_queries):
    return {
        "positives":  [ex.patch.id for ex in examples if ex.value],
        "negatives": [ex.patch.id for ex in examples if not ex.value],
        "queries": [pq.patch.id for pq in patch_queries]
    }
