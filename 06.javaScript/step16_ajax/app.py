from flask import Flask, request, render_template
from dao import EmpDAO

app = Flask(__name__)

#http://127.0.0.1:5000 >> http://127.0.0.1:5000/ 같은 표현~
#method속성 생략시 get방식 
@app.route('/', methods=['get'])
def get():
    print("get")
    return render_template('request.html')


@app.route('/getdata', methods=['get'])
def getdata():
    print("getdata()-------")
    return '{"name":"지은", "age":29}'

@app.route('/getemp', methods=['post'])
def dataemp():
    #???
    empno = request.form.get('empno')    ##????? 어떻게 client가 전송하는 데이터를 획득할 수 있을까?
    print("------------", empno)
    
    dao = EmpDAO()      #empone() 메소드를 보유한 객체 생성
    data = dao.empone(empno) #select 후에 json 포맷의 문자열로 가공해서 반환하는 메소드 호출
    return data

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port="5000")
