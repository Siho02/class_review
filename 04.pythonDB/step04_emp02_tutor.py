# oracle driver 모듈 사용 선언
import cx_Oracle


# emp01 table 생성
# create table emp01 as select empno, ename from emp
# step01 - 예외 처리 필요성 / step02 - 예외 처리 적용
# def emp01_create():
#     conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
#     cur = conn.cursor()
    
#     try:
#         cur.execute('drop table emp01')
#         cur.execute('create table emp01 as select empno, ename from emp')
#     except:
#         print('예외 발생')    
#     finally:
#         print('예외 발생 여부와 무관하게 100 실행되는 영역 - 자원 반환은 필수')
#         cur.close()
#         conn.close()


# def emp01_create():
#     conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
#     cur = conn.cursor()
    
#     try:
#         cur.execute('drop table emp01')
#     except:
#         print('table 미 존재')   # pass
    
#     try:
#         cur.execute('create table-------------- emp01 as select empno, ename from emp')
#     except  DataBaseError:
#         print("table 생성시 오류")

#     cur.close()
#     conn.close()


# sql문장별 발생 가능한 예외 처리를 try~except 라는 예외처리 문장 하나로 처리
# Exception as e 표현으로 발생되는 상황 인지 했음
# def emp01_create():
#     conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
#     cur = conn.cursor()
    
#     try:
#         cur.execute('drop table emp01')
#         cur.execute('create table ------- emp01 as select empno, ename from emp')
#     except Exception as e:
#         print(e)
#     finally:
#         cur.close()
#         conn.close()

def emp01_create():
    conn = cx_Oracle.connect(user="SCOTT22", password="TIGER", dsn="xe")
    cur = conn.cursor()
    
    try:
        cur.execute('drop table emp01')
        cur.execute('create table emp01 as select empno, ename from emp')
    except Exception as e:
        print(e)
    finally:
        cur.close()
        conn.close()





# ?
# emp02 query - all select / one select ...
# select 문장 : 필요로 데이터 있을경우 parameter / select * : 무
def emp01_query_all():
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()
    cur.execute("select * from emp01")

    # fetchall() : cursor로 부터 모든 검색결과를 반환 하는 함수
    # rows는 resetset 보유
    rows = cur.fetchall()

    for row in rows:
        print(row)

    cur.close()
    conn.close()


# ? empno로 해당 사원의 이름을 검색 
def emp01_query_one(empno):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()

    # 단 test용 emp01 table은 아쉽지만 복제 이 경우 alter명령어로 제약조건 추가하지 않은 상황
    # empno사번 중복되어 insert 되는 상황 
    # 검색된 결과가 row 하나 
    cur.execute("select * from emp01 where empno=:empno", empno=empno)

    # 결과가 하나의 row 
    row = cur.fetchone()
    print(row)

    cur.close()
    conn.close()


# emp02 insert
# ? insert into emp01 values (?, ?)
# parameter
# 가변적인 데이터 적용을 위한 binding 변수라는 사용 법 추가 학습 
def emp01_insert(new_empno, new_ename):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()
    # emp01_insert(22, "유재석")

    # step01 - 실행시 문법 오류 
    # cur.execute("insert into emp01 values(:new_empno, :new_ename)", new_empno, new_ename)

    # step02 - 메뉴얼 처럼 값을 변수에 직접 대입 즉 binding 변수에 값 대입
    # cur.execute("insert into emp01 values(:new_empno, :new_ename)", new_empno=22, new_ename='유재석')

    # step03 - step02 단계 성공, 따라서 함수의 parameter로 유입되는 데이터값을 보유한 로컬 변수로 binding 
    cur.execute("insert into emp01 values(:new_empno, :new_ename)", new_empno=new_empno, new_ename=new_ename)
    conn.commit()

    cur.close()
    conn.close()



# emp01 update
# ? update emp01 set ename=? where empno=?
# parameter
def emp01_update(empno, new_ename):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()

    cur.execute("update emp01 set ename=:new_ename where empno=:empno", new_ename=new_ename, empno=empno)
    conn.commit()

    cur.close()
    conn.close()

# emp01 delete
# ? delete from emp01 where empno=?
# parameter
def emp01_delete(del_empno):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()

    cur.execute("delete from emp01 where empno=:empno", empno=del_empno)

    conn.commit()
    cur.close()
    conn.close()
    

# 참고 : 현업에선 DDL문장은 가급적 프로그램 코드로 하지 않고 sql문장으로 작업 권장
# table 구조 변경은 가급적 최소화

# 개발되어 있는 함수를 필요에 의해서 직접 코드로 호출
# python에서 실행 순서에 대한 제어 또는 python 파일을 독립적으로 실행할 때 필요한 코드
if __name__ == '__main__':
    # 호출 순서 : table 생성 -> 검색 -> 저장 -> 검색 -> 수정 -> 검색 -> 삭제 -> 검색 
    emp01_create()
    # emp01_insert(22, "유재석")
    # emp01_update(7369, '스미스')
    # emp01_delete(7369)
    # emp01_query_all()
    # print('--- 7902 검색 ---')
    # emp01_query_one(7902)


