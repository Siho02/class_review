import cx_Oracle  # oracle db를 쉽게 활용 가능하게 해주는 driver

# connect() : id/pw/dsn 값으로 db 접속


connection = cx_Oracle.connect(user="hr", password="hr", dsn="xe") #dsn은 db별칭?
print('db 접속 성공')

# 접속된 db에 sql문장 실행 및 결과값 활용 가능하게 해 주는 기능
cursor = connection.cursor()

# execute() : query(select, 질의) 문장 실행 가능한 함수 
cursor.execute("""
        SELECT first_name, last_name
        FROM employees
        WHERE department_id = :did AND employee_id > :eid""", 
        did = 50,
        eid = 190)

# fname : 성 / lname : 이름 / cursor
for fname, lname in cursor:
    print("Values:", fname, lname)

# 자원 반환 필수 
cursor.close()
connection.close()



'''
FIRST_NAME                               LAST_NAME
---------------------------------------- --------------------------------------------------
Randall                                  Perkins
Sarah                                    Bell
Britney                                  Everett
Samuel                                   McCain
Vance                                    Jones
Alana                                    Walsh
Kevin                                    Feeney
Donald                                   OConnell
Douglas                                  Grant

'''
