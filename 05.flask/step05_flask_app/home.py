from flask import Flask, render_template
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/join_us', methods = ['post'])
def join_us():
    return render_template('join_us.html')

@app.route('/login', methods = ['post'])
def login():
    id = request.form.get('id')
    info = {"name" : "taeik", "age" : 30}
    return render_template('login.html', idencore=id, userinfo = info)

@app.route('/join', methods = ['post'])
def join():
    id = request.form.get('id')
    return render_template('welcome_join.html', idencore = id)


if __name__ == '__main__':
    app.run(debug=True)