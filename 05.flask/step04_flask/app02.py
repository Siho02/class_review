# html 로 응답 지시()

from flask import Flask, render_template
# get? post? 방식 구분 api
from flask import request

app = Flask(__name__)


# http://127.0.0.1:5000/ 로 직접 브라우저에서 요청 - get방식(default)
# post & get 방식 적용
# post 방식 test - post man 또는 html form에서 post로 지정한 tag로 요청
@app.route('/', methods=['post', 'get'])
def index():
    print('요청 방식(method) 확인 -- ', request.method)
    # html page 실행 즉 렌더링 지시
    # templates 하위의 html 파일 실행 시 teamplates 폴더명은 절대 코드에 적용하지 않음
    return render_template('step02_response.html')


if __name__ == '__main__':
    app.run(debug=True)
