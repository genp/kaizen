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

u = User.ifNew(username="Darius")
if u:
    u.password = user_manager.hash_password("dariuspass")
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
    ps.width = 200
    ps.height = 200
    ps.xoverlap = 0.75
    ps.yoverlap = 0.75
    ps.scale = 1.5
    ps.fliplr = True
    db.session.add(ps)



fs = FeatureSpec.ifNew(name = 'RGB', cls = 'extract.ColorHist')
if fs:
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'RGB coarse', cls = 'extract.ColorHist')
if fs:
    fs.params = {'bins': 3}
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'CNN_VGG', cls = 'extract.CNN')
if fs:
    fs.params = {'model':'VGG', 'layer_name': 'fc7'}
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'CNN_CaffeNet', cls = 'extract.CNN')
if fs:
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'HoG Dalal_Triggs', cls = 'extract.HoGDalal')
if fs:
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'TinyImage', cls = 'extract.TinyImage')
if fs:
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
