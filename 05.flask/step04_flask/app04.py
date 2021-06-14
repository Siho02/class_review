from flask import Flask, render_template

# client가 요청시에 서버에 전송하는 데이터를 활용할 수 있는 API
from flask import request

app = Flask(__name__)

#첫 요청시 step04_request.html로 렌더링
#get 방식
@app.route('/')
def index():
    return render_template('step04_request.html')

#step04_requst.html의 form tag에 입력된 데이터를 받아서 처리하는 함수
#http://127.0.0.1:5000/idcheck
#post 방식을 ㅗ처리
@app.route('/login', methods=['post'])
def login():
    #<input type = "text" id="id" name="id" value="test"
    #bane 속성값이 id인 tag의 value값을 획득하는 로직
    ## print('----', request.form.get('id'))

    '''http://127.0.0.1 >> app04.py의 get방식으로 index()실행 >> step04_request.html >> id/pw입력 
        >> 로그인 버튼 클릭(action = 'login' method = 'post') 
        >> url http://127.0.0.1/login으로 id/pw를 은닉해서 post방식으로 전송
        >> app04.py의 login url과 매핑도 login() 함수 실행
        >> 입력된 데이터값 확인(request.form.get('id'))
        >> 다음 html에서 출력하게끔 데이터를 전송 
        >> 응답하는 step04_response.html에선 app04.py가 전송한 데이터값을 출력하는 기능의 코드 구현
    '''

    id = request.form.get('id')

    #client가 입력된 데이터값을 받아서 step04_response.html에서 출력
    #로직 : app04.py에서 html로 데이터 전송

    #서버에서 새로운 데이터를 새로 구성하여 출력 담당 코드로 데이터 전달
    #데이터 구분을 위해 key와 value가 mapping된 구조인 json 포맷 사용(권장)
    info = {"name" : "taeik", "age" : 30}

    # idencore = id, userinfo = info / idencore와 userinfo는 데이터 구분용 key
    return render_template('step04_response.html', idencore=id, userinfo = info)

if __name__ == '__main__':
    app.run(debug=True)