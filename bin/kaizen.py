#!/usr/bin/env python
from app import app, db, user_manager
import config
from werkzeug.wsgi import DispatcherMiddleware

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

ps = PatchSpec.ifNew(name='CamelyonLvl2')
if ps:
    ps.width = 256
    ps.height = 256
    ps.xoverlap = 0
    ps.yoverlap = 0
    ps.scale = 200000
    ps.fliplr = False
    db.session.add(ps)

ps = PatchSpec.ifNew(name='MedDense')
if ps:
    ps.width = 200
    ps.height = 200
    ps.xoverlap = 0.5
    ps.yoverlap = 0.5
    ps.scale = 2
    ps.fliplr = True
    db.session.add(ps)

ps = PatchSpec.ifNew(name='LessDense_NoFlip')
if ps:
    ps.width = 200
    ps.height = 200
    ps.xoverlap = 0.25
    ps.yoverlap = 0.25
    ps.scale = 2
    ps.fliplr = True
    db.session.add(ps)

ps = PatchSpec.ifNew(name='Dense')
if ps:
    ps.width = 200
    ps.height = 200
    ps.xoverlap = 0.75
    ps.yoverlap = 0.75
    ps.scale = 3
    ps.fliplr = True
    db.session.add(ps)


fs = FeatureSpec.ifNew(name = 'RGB', cls = 'extract.ColorHist')
if fs:
    db.session.add(fs)

fs = FeatureSpec.ifNew(name = 'RGB coarse', cls = 'extract.ColorHist')
if fs:
    fs.params = {'bins': 3}
    db.session.add(fs)


# fs = FeatureSpec.ifNew(name = 'CNN_VGG', cls = 'extract.CNN')
# if fs:
#     print "adding CNN_VGG FeatureSpec"
#     fs.params = {'model':'VGG', 'layer_name': 'fc7'}
#     db.session.add(fs)
#     print 'loaded CNN_VGG FeatureSpec'

# fs = FeatureSpec.ifNew(name = 'CNN_VGG_redux', cls = 'extract.CNN')
# if fs:
#     print "adding CNN_VGG FeatureSpec with power norm and 200D"
#     fs.params = {'model':'VGG', 'layer_name': 'fc7', 'use_reduce' : True, 'ops' : ["subsample", "power_norm"], 'output_dim' : 200, 'alpha' : 2.5}
#     db.session.add(fs)
#     print 'loaded CNN_VGG redux FeatureSpec'


# fs = FeatureSpec.ifNew(name = 'CNN_CaffeNet', cls = 'extract.CNN')
# if fs:
#     print "adding CNN_CaffeNet FeatureSpec"
#     db.session.add(fs)
#     print "loaded CNN_CaffeNet FeatureSpec"

fs = FeatureSpec.ifNew(name = 'CNN_CaffeNet_redux', cls = 'extract.CNN')
if fs:
    print "adding CNN_CaffeNet redux FeatureSpec"
    fs.params = {'use_reduce' : True, 'ops' : ["subsample", "power_norm"], 'output_dim' : 200, 'alpha' : 2.5}
    db.session.add(fs)
    print "loaded CNN_CaffeNet redux FeatureSpec"

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

e = Estimator.ifNew(cls = 'sklearn.svm.LinearSVC')
if e:
    e.params = {}
    db.session.add(e)

db.session.commit()

def simple(env, resp):
    resp(b'200 OK', [(b'Content-Type', b'text/plain')])
    return [b'If you see this, unwrap the app in kaizen.py']

app.wsgi_app = DispatcherMiddleware(simple, {'/kaizen': app.wsgi_app})

flask_host = config.HOST
# Listen to outside connections if we have a "real" hostname
if flask_host != "localhost":
    flask_host = "0.0.0.0"
if __name__ == '__main__':
    app.run(port=config.PORT, threaded=True)
