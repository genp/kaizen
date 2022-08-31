from app import app, db
from flask import render_template, redirect, url_for, jsonify, send_file
from app.forms import (
    ActiveQueryForm,
    ClassifierForm,
    BlobForm,
    DetectForm,
    ClassifierEvaluateForm,
)
from app.models import (
    User,
    Classifier,
    PatchQuery,
    PatchResponse,
    HitResponse,
    Estimator,
    Detection,
    Round,
    Dataset,
    ValPrediction,
    Blob,
    Patch,
    dataset_x_blob,
)


import tasks

import time, itertools

# TODO: this should be in config.py
SAMPLE_LIMIT = 10000

@app.route("/classifier/")
def classifier_top():
    classifiers = (
        Classifier.query.filter_by(is_ready=True)
        .order_by(Classifier.id.desc())
        .limit(50)
    )
    return render_template(
        "classifier_top.html",
        title="Classifier Library",
        classifiers=classifiers,
        next=next,
    )


@app.route("/classifier/<int:id>")
def classifier(id, r_id=None):
    classifier = Classifier.query.get_or_404(id)

    if r_id is not None:
        round = Round.query.filter(
            Round.classifier == classifier, Round.number == r_id
        ).first()
    else:
        round = classifier.latest_round

    hits = make_hit(classifier.rounds[0].examples, round.queries)

    form = ActiveQueryForm()
    form.classifier.data = classifier
    form.round.data = round
    classifier_title = (
        classifier.keyword.name if classifier.keyword else "Export Classifier"
    )

    return render_template(
        "classifier.html",
        classifier=classifier,
        round=round,
        title="%s %d" % (classifier_title, classifier.rounds.count()),
        form=form,
        hits=hits,
    )


@app.route("/classifier/<int:id>/<int:i>/")
def classifier_round(id, i):
    return classifier(id, r_id=i)

    # return render_template( 'classifier_round.html', classifier=classifier,
    #                        title='%s %d' % (classifier_title, i),
    #                        form=form, round = round, hits = hits)


@app.route("/classifier/<int:id>/download")
def classifier_dl(id):
    classifier = Classifier.query.get_or_404(id)
    round = classifier.latest_round
    return send_file(round.location)


@app.route("/classifier/update/<int:id>", methods=["POST"])
def classifier_update(id):
    form = ActiveQueryForm()

    if form.validate_on_submit():
        print("form.user.data: " + form.user.data)
        user = None  # current_user

        if True:  # user.is_anonymous and form.user.data:
            # TODO fix when authentication is fixed
            user = User.query.get(1)  # User.find(form.user.data)
            # if user:
            #     assert user.password == None
            # else:
            #     user = User(username=form.user.data, password=None)
            #     db.session.add(user)
            #     user.location = form.location.data
            #     user.nationality = form.nationality.data
        print("update classifier %s from %s....." % (id, user))

        hit_resp = HitResponse(
            time=form.time.data, user=None if user.is_anonymous else user
        )
        db.session.add(hit_resp)
        db.session.commit()
        print(str(time.time()) + ": " + str(hit_resp))

        pos_response = [(p, True) for p in form.pos_patches.data]
        neg_response = [(p, False) for p in form.neg_patches.data]

        for patch, v in itertools.chain(pos_response, neg_response):
            pq = PatchQuery.query.filter_by(patch=patch, round=form.round.data).one()
            pr = PatchResponse(value=v, hitresponse=hit_resp, patchquery=pq)
            db.session.add(pr)

        print(str(time.time()) + ": done")

        db.session.commit()
        print(str(time.time()) + ": committed")
        tasks.advance_classifier.delay(form.classifier.data.id, limited_number_of_features_to_evaluate=SAMPLE_LIMIT)

    else:
        for field in form:
            print(field.name)
            print(field.errors)
            print(field.data)
    return jsonify(results="success")


@app.route("/classifier/new", methods=["POST"])
def classifier_new():
    form = ClassifierForm()

    if form.validate_on_submit():
        c = Classifier(
            dataset=form.dataset.data,
            keyword=form.keyword.data,
            estimator=form.estimator.data,
        )
        db.session.add(c)
        db.session.commit()

        # TODO: for now, a new classifier randomly samples at most 50k features; consider parameterizing this.
        tasks.if_classifier(c, limited_number_of_features_to_evaluate=SAMPLE_LIMIT)
        return redirect(c.url)
    else:
        print("did not validate")
        print(form.dataset.errors)
        return redirect(url_for("classifier_top"))


@app.route("/classifier/<int:classifier_id>/<int:round_id>/evaluate/<int:dataset_id>")
def classifier_evaluate(classifier_id, round_id, dataset_id):
    classifier = Classifier.query.get_or_404(classifier_id)
    round = Round.query.get_or_404(round_id)
    dataset = Dataset.query.get_or_404(dataset_id)

    form = ClassifierEvaluateForm()
    form.classifier.data = classifier
    form.round.data = round
    form.dataset.data = dataset

    if dataset.is_train:
        return "not validation set"

    # maybe instead I can check the notes in the round
    predicts = db.engine.execute(
        "SELECT * from val_prediction where patch_id in\
                            (select id from patch where blob_id in \
                            (select blob_id from dataset_x_blob where dataset_id=%d)) \
                            and round_id=%d\
                            ORDER BY value DESC;"
        % (dataset.id, round.id)
    ).fetchmany(100)
    if predicts:
        first_last_patches = None
        form.note.data = None
        if round.notes:
            rn = eval(round.notes)
            for note_id in rn:
                user_id = -1  # if current_user.is_anonymous else current_user.id
                if (
                    rn[note_id]["dataset"] == dataset.id
                    and rn[note_id]["user"] == user_id
                ):
                    first_last_patches = rn[note_id]["first_last_patches"]
                    form.note.data = note_id
        return render_template(
            "classifier_evaluate.html",
            predicts=predicts,
            form=form,
            eval=first_last_patches,
        )
    return redirect(url_for("evaluate_top"))


@app.route("/classifier/eval_range", methods=["POST"])
def classifier_eval_range():
    form = ClassifierEvaluateForm()
    if form.validate_on_submit():
        user = None  # current_user
        round = form.round.data
        round = Round.query.get(round.id)

        dataset = form.dataset.data
        first_incorrect = form.first_incorrect.data
        last_correct = form.last_correct.data
        note = form.note.data

        prev_notes = eval(round.notes) if round.notes else [0]

        if note:
            note = int(note)
            tops_bottoms = {
                "eval_score": abs(first_incorrect.id - last_correct.id),
                "first_last_patches": (first_incorrect.id, last_correct.id),
            }
            new_note = {**prev_notes[note], **tops_bottoms}
            round.notes = str({**prev_notes, **{note: new_note}})
        else:
            # pass
            insert_id = -1 if user.is_anonymous else user.id
            note = max(prev_notes) + 1
            tops_bottoms = {
                note: {
                    "user": insert_id,
                    "dataset": dataset.id,
                    "eval_score": abs(first_incorrect.id - last_correct.id),
                    "first_last_patches": (first_incorrect.id, last_correct.id),
                }
            }
            round.notes = (
                str({**prev_notes, **tops_bottoms})
                if round.notes
                else str(tops_bottoms)
            )
        db.session.commit()
        return jsonify({"note_id": note})
    else:
        return redirect(url_for("evaluate_top"))


####################
### Detect Views ###
####################
@app.route("/detect/", methods=["GET", "POST"])
def detect_top():
    form = BlobForm()
    detect_form = DetectForm()
    if detect_form.validate_on_submit():
        blobs = detect_form.blobs.data
        detects = []
        for blob in blobs:
            d = Detection(blob=blob)
            db.session.add(d)
            detects.append(d)
        db.session.commit()
        for d in detects:
            tasks.detect.delay(d.id)

    return render_template(
        "detect.html",
        title="Detect",
        form=form,
        detect_form=detect_form,
        detects=Detection.query.all(),
    )


@app.route("/detect/<int:id>/")
def detect(id):
    detect = Detection.query.get_or_404(id)

    return render_template(
        "detection.html", detect=detect, title="Detect %d" % (detect.id)
    )


def make_hit(examples, patch_queries):
    return {
        "positives": [ex.patch.id for ex in examples if ex.value],
        "negatives": [ex.patch.id for ex in examples if not ex.value],
        "queries": [pq.patch.id for pq in patch_queries],
    }
