#!/usr/bin/env python
from flask import Flask
from flask.ext.assets import Environment
from flask.ext.sqlalchemy import SQLAlchemy
from flask_user import UserManager, SQLAlchemyAdapter
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
from config import APPNAME, LOG_FILE

app = Flask(__name__, static_url_path='')
app.config.from_object('config')

assets = Environment(app)
db = SQLAlchemy(app)

from app import views, models
user_manager = UserManager(SQLAlchemyAdapter(db, models.User), app)

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
