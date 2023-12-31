from flask import Flask, jsonify, send_file, session, \
    request, render_template, send_from_directory
from flask_cors import *
from flask_session import Session
from datetime import timedelta
from .views.admin import admin

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r'/*': {"origins": "*"}})
    app.config['JSON_SORT_KEYS'] = False
    app.config["SECRET_KEY"] = b'\xf4S\xef2R&\x06\xbd\xf0\xf3\xb5\x86o\xca\x95\x14\x8e\x0f\x8c\xd3;\\S6'
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    Session(app)

    # app.config.from_object("config.DevelopmentConfig")

    app.register_blueprint(admin)
    # # this kind of manner is somewhere wrong
    # app.config.update(
    #     DEBUG=True,
    #     SERVER_NAME = "0.0.0.0:8000"
    # )

    return app