#!/usr/bin/env python
from app import app, db, user_manager

# Preload the DB so we can drop it any time.
from app.models import User, PatchSpec, FeatureSpec, Estimator
u = User.ifNew(username="jj")
if u:
    u.password = user_manager.hash_password("jjpass")
    u.is_enabled = True;
    db.session.add(u)

u = User.ifNew(username="gen")
if u:
    u.password = user_manager.hash_password("genpass")
    u.is_enabled = True;
    db.session.add(u)


ps = PatchSpec.ifNew(name='Sparse')
if ps:
    ps.width = 400
    ps.height = 400
    ps.xoverlap = 0.5
    ps.yoverlap = 0.5
    ps.scale = 3.0
    db.session.add(ps)

ps = PatchSpec.ifNew(name='Dense')
if ps:
    ps.width = 300
    ps.height = 300
    ps.xoverlap = 0.6
    ps.yoverlap = 0.6
    ps.scale = 2.0
    ps.fliplr = True
    db.session.add(ps)


fs = FeatureSpec.ifNew(name = 'RGB', cls = 'extract.ColorHist')
if fs:
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'RGB coarse', cls = 'extract.ColorHist')
if fs:
    fs.params = {'bins': 3}
    db.session.add(fs)

e = Estimator.ifNew(cls = 'sklearn.neighbors.KNeighborsRegressor')
if e:
    e.params = {'weights' : 'distance', 'n_neighbors': 2}
    db.session.add(e)

e = Estimator.ifNew(cls = 'sklearn.linear_model.LinearRegression')
if e:
    e.params = {'normalize' : True, 'n_jobs' : 2}
    db.session.add(e)

db.session.commit()

app.run(host='localhost', port=8080)
