#!/usr/bin/env python
import os
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
from config import APPNAME, log_file, log_path
from flask.ext.assets import Environment, Bundle

app = Flask(__name__, static_url_path='')
app.config.from_object('config')

assets = Environment(app)
db = SQLAlchemy(app)

from app import views

# email logs for catasrophic failures 
from config import clroot, ADMINS, MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD

credentials = None
if MAIL_USERNAME or MAIL_PASSWORD:
    credentials = (MAIL_USERNAME, MAIL_PASSWORD)
mail_handler = SMTPHandler((MAIL_SERVER, MAIL_PORT), 'no-reply@' + MAIL_SERVER, ADMINS, APPNAME+' failure', credentials)
mail_handler.setLevel(logging.ERROR)
app.logger.addHandler(mail_handler)

# log for app events
if not os.path.exists(log_path):
    os.mkdir(log_path)
    
file_handler = RotatingFileHandler(log_file, 'a', 1 * 1024 * 1024, 10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
