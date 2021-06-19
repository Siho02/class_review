
/* json 포맷의 문자열 >> json객체로 변환
demo의 위치에 table을 생성해서 직원 수 만큼 row를 구성해야..
table의 row tag는 동일
    반복문 + 데이터만 가변적으로 json으로부터 뽑음
        <tr><td>사번</td><td>이름</td><td></td>
*/

//https://poiemaweb.com/es6-template-literals

function empall(){
    const xhttp = new XMLHttpRequest(); 
    xhttp.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            data = this.responseText;
            data = JSON.parse(data);    

            tab = `
            <table border="1">
                <tr><th>사번</th><th>이름</th><th>급여</th></tr>`;
            
            let empno;
            let enmae;
            let sal;

            for(no in data){
                empno = data[no].empno;
                ename = data[no].ename;
                sal = data[no].sal; 
                tab = tab + `<tr>
                    <td>${empno}</td>
                    <td>${ename}</td>
                    <td>${sal}</td>
                </tr>`;
            }
            tab = tab + `</table>`;
            document.getElementById("demo").innerHTML = tab;
            //console.log(tab);
            
        };
    };
    xhttp.open("GET", "emplist");    
    xhttp.send();
}
    
