from flask import Flask, request, render_template
from dao import bank_get2

app = Flask(__name__)

#index.html 단순 실행
@app.route("/", methods=["get"])
def index_view():
    return render_template("index_tutor.html")

#버튼 클릭시 비동기로 응답
@app.route("/dataget", methods=["get"])
#def req_res():
#    data = bank_get2()
#    print(dict(data))
#    return dict(data)

## 응답된 데이터 첫번째 형식
# js에서 문제 발생 >> eval() & parse() 모두 실패
    # JSON.parse(문자열) : key와 value의 구조는 큰따옴표 표기 필수 >> json 객체로 변환
    # eval("(" + data ")") >> json 객체로 변환
# 해결책 : {},{} 여러개인 경우 >> json 배열 형식어야 함. json 배열 형식의 문자열로 변환시도
# 첫번째 방식
# 두번째 방식
def req_res():
    data = bank_get2()
    # print(data)
    str_data = ""
    for i in range(len(data)):
        str_data += str(data[i]) + ","
    # print("-------", type(str_data)) 
    return str_data


if __name__ == "__main__":
    app.run(debug=True, host = "127.0.0.1")