#!/usr/bin/env python
from flask import Flask
from flask_assets import Environment
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
#from flask_user import UserManager, SQLAlchemyAdapter
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
from config import APPNAME, LOG_FILE

app = Flask(__name__, static_url_path='')
app.config.from_object('config')

assets = Environment(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# DO NOT move this import to the top of this file.
# It will cause a circular import.
from app import views, models
# user_manager = UserManager(SQLAlchemyAdapter(db, models.User), app)
@login_manager.user_loader
def load_user(user_id):
    return models.User.query.get(user_id)
login_manager.login_view = "user.login"

from flask_login import AnonymousUserMixin
class Guest(AnonymousUserMixin):

    def can(self, permission_name):
        return False

    @property
    def is_admin(self):
        return False

login_manager.anonymous_user = Guest

# convenience function for jinja to render app name in templates
@app.context_processor
def get_appname():
    return dict(appname=APPNAME)

# email logs for catasrophic failures
from config import ADMINS, MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD

credentials = None
if MAIL_USERNAME or MAIL_PASSWORD:
    credentials = (MAIL_USERNAME, MAIL_PASSWORD)
mail_handler = SMTPHandler((MAIL_SERVER, MAIL_PORT), 'no-reply@' + MAIL_SERVER, ADMINS, APPNAME+' failure', credentials)
mail_handler.setLevel(logging.ERROR)
app.logger.addHandler(mail_handler)

file_handler = RotatingFileHandler(LOG_FILE, 'a', 1 * 1024 * 1024, 10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)


# adding support for converting numpy types to sql
import numpy
from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
  return AsIs(numpy_float64)
register_adapter(numpy.float64, addapt_numpy_float64)

def addapt_numpy_float32(numpy_float32):
  return AsIs(numpy_float32)
register_adapter(numpy.float32, addapt_numpy_float32)
