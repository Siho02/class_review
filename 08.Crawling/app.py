import time
from bs4 import BeautifulSoup
from flask.json import jsonify
from selenium import webdriver
from flask import Flask, render_template, request
import requests
from google_search import search_from_google

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['get'])
def index():
    return render_template('index.html')

@app.route('/box', methods=["GET"])
def box():
    x = search_from_google()
    # print(x)
    return jsonify(x)


if __name__ == '__main__':
    app.run(debug=True)