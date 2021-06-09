## dept table : deptno는 절대 중복 불허
    ## 1. CRUD 로직만 구현
    ## 2. step05_dept_crud.py : 예외 처리 꼼꼼하게 구현

import cx_Oracle

def dept01_create():
    try :
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
        try:
            cur = conn.cursor()
            cur.execute('drop table dept01')
        except Exception as e:
            print('exception OCCUR : ', e)
        finally:
            cur.execute('create table dept01 as select deptno, dname, loc from dept')
            cur.execute('alter table dept01 add constraint pk_dept01_deptno primary key (deptno)')
            
    except Exception as e:
        print('exception OCCUR : ', e)
    finally:
        cur.close()
        conn.close()

def dept01_insert(new_deptno, new_dname, new_loc):
    try:
        conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
        cur = conn.cursor()
        try:
            cur.execute("insert into dept01 values(:new_deptno, :new_dname, :new_loc)", 
                    new_deptno=new_deptno, new_dname=new_dname, new_loc=new_loc)
        conn.commit()
    except Exception as e:
        print('exception occur : ', e)
    finally:
        cur.close()
        conn.close()

def dept01_update(deptno, new_dname, new_loc):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()
    
    try:
        cur.execute("update dept01 set dname=:new_dname, loc=:new_loc where deptno =:deptno",
                    deptno = deptno ,new_dname = new_dname, new_loc = new_loc)
        conn.commit()
    except Exception as e:
        print('exception occur : ', e)
    finally:
        cur.close()
        conn.close()

def dept01_delete(del_deptno):
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor()
    
    try:
        cur.execute("delete from dept01 where deptno=:del_deptno", del_deptno = del_deptno)
        conn.commit()
    except Exception as e:
        print('exception occur : ', e)
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    dept01_create()
    dept01_insert(50, 'analyzing', 'new_york')
    dept01_update(10, 'account', 'new_york')
    dept01_delete(20)