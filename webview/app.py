import os
from flask import Flask
from page import cifar_page
from datetime import timedelta

app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

app.register_blueprint(cifar_page)