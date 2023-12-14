from flask import Blueprint, render_template, abort, session, send_file, request
import os

admin = Blueprint("admin", __name__)

@admin.route("/")
def index():
    session["logged_in"] = True
    # get model here

    return render_template("index.html")


@admin.route("/GetManifest", methods=["GET", "POST"])
def app_get_manifest():
    # extract info from request
    return {"name": "backend"}
