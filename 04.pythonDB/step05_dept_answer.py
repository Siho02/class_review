"""1. crud 로직만 구현 create raed update delete
2. step05_dept_crud.py
dept table : deptno는 절대 중복 불허
예외처리를 꼼꼼하게 구현"""
import cx_Oracle
def dept01_create():
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe") # sql scott 계정 접속
        cur = conn.cursor() # Connection 객체로부터 cursor() 메서드를 호출
        try:
            cur.execute('drop table dept01') # execute : query 문장 실행 가능한 함수, dept01 drop
        except Exception as e: # 예외가 발생 했을 경우, 오류 및 충돌방지..;
            print(e)
        cur.execute('create table dept01 as select * from dept')# dept01 테이블 생성, 내용은dept *
        cur.execute('alter table dept01 add constraint dept01_ukl_deptno unique(deptno)') # deptno 중복 불허 조건을 위해 unique 필드 생성
    except Exception as e: # 에러가 날 경우 print 'e'
        print(e)
    finally:
        cur.close() # 자원 반환
        conn.close()

def dept01_query(find_dname):  # 찾고자 하는 값 검색
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
        cur = conn.cursor()
        try:
            cur.execute("select * from dept01 where dname like :dname", dname=find_dname) # like 특정 데이터 추출,WHERE열의 특정 패턴을 검색
            rows = cur.fetchall() # rows data를 한꺼번에 클라이언트로 가져옴
            for row in rows: # row in rows 반복문 
                print(row)
        except Exception as e: # 예외..
                print(e) # print e
    except Exception as e: # 
        print(e)
    finally:
        cur.close() # 자원 반환
        conn.close()
def dept01_insert(new_deptno, new_dname, new_loc): # 새로운 값 추가 insert
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe") 
        try:
            cur = conn.cursor()
            cur.execute("insert into dept01 values(:deptno, :dname, :loc)",\
            deptno=new_deptno, dname=new_dname, loc=new_loc) # deptno, dname, loc Value로 new값 생성
            conn.commit() # 영구 저장
        except Exception as e: # 에러가 날 경우 print 'e'
            print(e)
        finally:
            cur.close()
            conn.close()
    except Exception as e: # 에러가 날 경우 print 'e'
        print(e)
def dept01_update(deptno, new_dname, new_loc): # 테이블에 있는 데이터 수정 update
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe") # 접속
        try:
            cur = conn.cursor()
            cur.execute("update dept01 set dname=:dname, loc=:loc where deptno=:deptno",\
            dname=new_dname, loc=new_loc, deptno=deptno) # dept01.deptno 조건 dname, loc update 
            conn.commit() # 저장
        except Exception as e:  # 에러가 날 경우 print 'e'
            print(e) 
        finally:
            cur.close() # 자원 반환
            conn.close()
    except Exception as e:
        print(e)
def dept01_delete(deptno): # 데이터 삭제 delete
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe") # 접속
        try:
            cur = conn.cursor()
            cur.execute("delete from dept01 where deptno=:deptno", deptno=deptno) # deptno 값 대입하여 delete
            conn.commit() # 저장
        except Exception as e:
            print(e)
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        print(e)
        
if __name__ == '__main__': # 메인함수 선언
    dept01_create()
    #dept01_insert(50, 'PD', '강남')
    #dept01_query('%ING') # dname 값 추출
    #dept01_update(50, 'Playdata','남터')
    #dept01_query('P%')  
    #dept01_delete(50)
    #dept01_query('P%')
    #dept01_query()
# ORA-00942: table or view does not exist 오류발생 (기존에 dept01 테이블이 없었을 경우 생성 됐음..)
# 해결방법 ; 
# GRANT SELECT ON [dept01]] TO [USER];
# GRANT INSERT ON [dept01] TO [USER];
# GRANT DELETE ON [dept01] TO [USER];
# GRANT UPDATE ON [dept01] TO [USER];
# -- 또는
# GRANT SELECT, INSERT, DELETE, UPDATE ON [TABLE NAME] TO [USER]