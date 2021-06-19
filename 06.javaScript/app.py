from flask import Flask, request, render_template
from dao import EmpDAO
from dto import EmpDTO

app=Flask(__name__)


@app.route("/", methods=["get"])
def get():
    return render_template("reqres.html")   

@app.route("/getdata", methods=["get"])
def getdata():
    return '{"name":"재석", "age":49}'

@app.route("/getemp", methods=["post"])
def getemp():
    return EmpDAO().empone(request.form.get('empno'))

@app.route("/insert", methods=["post"])
def insertemp():
    dao = EmpDAO()
    dto = EmpDTO(request.form.get("empno"), request.form.get("ename"), request.form.get("sal"))
    dao.empinsert(dto)

    return dao.empone(request.form.get('empno'))


@app.route("/update", methods=["post"])
def updateemp():
    dao = EmpDAO()
    dto = EmpDTO(request.form.get("empno"), request.form.get("ename"), request.form.get("sal"))
    dao.empupdate(dto)

    return dao.empone(request.form.get('empno'))


@app.route("/delete", methods=["post"])
def deleteemp():
    dao = EmpDAO()
    empno = EmpDTO(request.form.get("empno"), request.form.get("ename"), request.form.get("sal"))
    dao.empdelete(empno)

    return dao.empone(request.form.get('empno'))

#모든 직원 정보를 요청 및 응답하는 함수
@app.route('/emplist', methods=["get"])
def emplist():
    print('-------')
    return EmpDAO().empall()



if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port="5000")