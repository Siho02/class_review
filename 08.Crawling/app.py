import time
from bs4 import BeautifulSoup
from flask.json import jsonify
from selenium import webdriver
from flask import Flask, render_template, request
import requests
from rank_crawl import crawl
from visual import visualization

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/box', methods=["GET"])
def box():
    x = crawl()
    return jsonify(x)

@app.route('/chart', methods=['GET'])
def chart():
    x = visualization()


if __name__ == '__main__':
    app.run(debug=True)