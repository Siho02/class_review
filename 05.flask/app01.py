from flask import Flask

# Falask 객체(instance) 생성
app = Flask(__name__)

# url 설정 - http://ip:port/ 또는 http://ip:port 형식으로 구성
# @ : 장식자
@app.route('/playdata')
def index():
    return '{"name" : "조태익"}'

if __name__ == '__main__':
    #Flask로 실행하기 위한 필수 코드
    #debug = True : server가 실행중이더라도 소스 수정시 >> 자동 갱신 가능
    app.run(debug=True)