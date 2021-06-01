-- 5.join.sql

/*
1. 조인이란?
	다수의 table간에  공통된 데이터를 기준으로 검색하는 명령어
	다수의 table이란?
		동일한 table을 논리적으로 다수의 table로 간주
			- self join
		물리적으로 다른 table간의 조인

2. 사용 table 
	1. emp & dept 
	  : deptno 컬럼을 기준으로 연관되어 있음

	 2. emp & salgrade
	  : sal 컬럼을 기준으로 연관되어 있음

  
3. table에 별칭 사용 
	검색시 다중 table의 컬럼명이 다를 경우 table별칭 사용 불필요, 
	서로 다른 table간의 컬럼명이 중복된 경우,
	컬럼 구분을 위해 오라클 엔진에게 정확한 table 소속명을 알려줘야 함
	- table명 또는 table별칭
	- 주의사항 : 컬럼별칭 as[옵션], table별칭 as 사용 불가


4. 조인 종류 
	1. 동등 조인
		 = 동등비교 연산자 사용
		 : 사용 빈도 가장 높음
		 : 테이블에서 같은 조건이 존재할 경우의 값 검색 

	2. not-equi 조인
		: 100% 일치하지 않고 특정 범위내의 데이터 조인시에 사용
		: between ~ and(비교 연산자)

	3. self 조인 
		: 동일 테이블 내에서 진행되는 조인
		: 동일 테이블 내에서 상이한 칼럼 참조
			emp의 empno[사번]과 mgr[사번] 관계

	4. outer 조인 
		: 두개 이상의 테이블이 조인될때 특정 데이터가 모든 테이블에 존재하지 않고 컬럼은 존재하나 null값을 보유한 경우
		  검색되지 않는 문제를 해결하기 위해 사용되는 조인
		  null 값이기 때문에 배제된 행을 결과에 포함 할 수 있드며 (+) 기호를 조인 조건에서 정보가 부족한 컬럼쪽에 적용
		
		: oracle DB의 sql인 경우 데이터가 null 쪽 table 에 + 기호 표기 */

-- 1. dept table의 구조 검색
desc dept;
-- dept, emp, salgrade table의 모든 데이터 검색
select * from dept;
select * from emp;
select * from salgrade;


 



--*** 1. 동등 조인 ***
-- = 동등 비교
-- 2. SMITH 의 이름(ename), 사번(empno), 근무지역(부서위치)(loc) 정보를 검색
select ename, empno, loc from emp, dept where ename='SMITH';

/*select ename, empno, loc, deptno from emp, dept where ename='SMITH';
	deptno는 두개의 테이블에 존재, 따라서 어떤 테이블의 어떤 컬럼인지 검색이 불명확
	ORA-00918
*/

select ename, empno, loc, emp.deptno, dept.deptno from emp, dept where ename='SMITH';

select ename, empno, loc from emp, dept where ename='SMITH' and emp.deptno = dept.deptno;

-- 3. deptno가 동일한 모든 데이터(*) 검색
-- emp & dept 
select * from emp, dept;
-- emp와 dept 테이블의 모든 정보가 출력은 되나 12 * 4 개 만큼 출력


-- 4. 2+3 번 항목 결합해서 SMITH에 대한 모든 정보(ename, empno, sal, comm, deptno, loc) 검색하기
select ename, empno, loc from emp, dept where ename='SMITH' and emp.deptno = dept.deptno;

-- 5.  SMITH에 대한 이름(ename)과 부서번호(deptno), 부서명(dept의 dname) 검색하기
select enmae, e.deptno, dname from emp e, dept where enmae = 'SMITH' and e.deptno = dept.deptno;

-- 6. 조인을 사용해서 뉴욕에 근무하는 사원의 이름과 급여를 검색 
-- loc='NEW YORK', ename, sal

-- 데이터가 매우 많다고 가정했을 때, 첫번째가 좀 더 선호하는 방식
select ename, sal from emp, dept where loc = 'NEW YORK' and emp.deptno = dept.deptno;
select ename, sal from emp, dept where emp.deptno = dept.deptno and loc = 'NEW YORK';

-- 7. 조인 사용해서 ACCOUNTING 부서(dname)에 소속된 사원의 이름과 입사일 검색
-- 소속된 사원(emp)의 이름(emp ename)과 입사일(emp hiredate)
select enmae, hiredate from emp, dept where dname = 'ACCOUNTING' and dept.deptno = emp.deptno


-- 8. 직급이 MANAGER인 사원의 이름, 부서명 검색
select ename, dname from emp, dept where job = 'MANAGER' and dept.deptno = emp.deptno;

-- 10. 사원(emp) 테이블의 부서 번호(deptno)로 부서 테이블을 참조하여 사원명, 부서번호, 부서의 이름(dname) 검색
select ename, emp.deptno, dname from emp, dept where emp.deptno = dept.deptno;

-- *** 2. not-equi 조인 ***

-- salgrade table(급여 등급 관련 table)
-- 9. 사원의 급여가 몇 등급인지 검색
-- between ~ and : 포함 
select * from selgrade;

select ename, sal, grade from emp, salgrade where sal between losal and hisal;


-- ?? : 81년 4월 1일 이후에 입사한 사원들이 가장 많은 부서의 부서명을 구하세요
-- dname의 개수를 카운팅해서 max 최대값
select ename, dname, hiredate from emp, dept where hiredate > '81/04/01' and emp.deptno = dept.deptno;
select distinct(dname), count(dname) from emp, dept where hiredate > '81/04/01' and emp.deptno = dept.deptno group by dname;

select 
from emp, dept
where hiredate > '81/04/01' and emp.deptno = dept.deptno 

-- 답은 나오는데 뭔가 이상?
select max(dname) from emp, dept where  hiredate > '81/04/01' and emp.deptno = dept.deptno;
select min(dname) from emp, dept where  hiredate > '81/04/01' and emp.deptno = dept.deptno;  
--

select * 
from (
	select dname
	from emp, dept
	where hiredate > '81/04/01' and emp.deptno = dept.deptno
	group by dname
	order by count(enmae) desc
	 )
where 

-- *** 3. self 조인 ***
-- 하나의 테이블에 연관된 컬럼들로 다수의 table인 듯 논리적으로 작업
-- 11. SMITH 직원(사원)의 메니저 이름(사원이 and 스미스 입장에서 상사) 검색
select m.ename 
from emp e, emp m 
where e.ename = 'SMITH' and e.mgr = m.empno;



-- 12. 메니저 이름이 KING(m ename='KING')인 사원들의 이름(e ename)과 직무(e job) 검색
--t 사원 테이블 = m , 매니저 테이블 = m
select e.ename, e.job from emp e, emp m
where m.ename = 'KING' and e.mgr = m.empno;



-- 13. SMITH와 동일한 근무지에서 근무하는 사원의 이름 검색
select you.ename
from emp my, emp you
where my.ename = 'SMITH' and my.deptno = you.deptno;

-- smith 제외하기
select you.ename
from emp my, emp you
where my.ename = 'SMITH' and my.deptno = you.deptno and you.ename != 'SMITH';
--where my.ename = 'SMITH' and my.deptno = you.deptno and not you.ename = 'SMITH';



--*** 4. outer join ***
/*
1. join은 ANSI join, 즉 RDBMS에 종속적이지 않은 표준 SQL 문장 공부 필요
2. 참고 링크 
	https://www.w3schools.com/sql/sql_join.asp
*/

-- 14. 모든 사원명, 메니저 명 검색, 단 메니저가 없는 사원도 검색되어야 함
-- deptno의 40은 존재 / 단, emp 테이블에는 deptno가 40인 직원 없음
-- emp의 KING은 mgr이 null
select e.ename as 사원명, m.ename as 매니저명
from emp e, emp m
where e.mgr = m.empno(+);

select e.ename as 사원명, m.ename as 매니저명
from emp e, emp m
where m.empno(+) = e.mgr;

/*
select e.ename as 사원명, m.ename as 매니저명
from emp e, emp m
where e.mgr(+) = m.empno;
>> 논리적으로 심각한 오류가 생김(코드상으로는 문제가 없음)
*/


-- 15. 모든 직원명(ename), 부서번호(deptno), 부서명(dname) 검색
-- 부서 테이블의 40번 부서와 조인할 사원 테이블의 부서 번호가 없지만,
-- outer join이용해서 40번 부서의 부서 이름도 검색하기 
select e.ename, d.deptno, d.dname
from emp e, dept d
where e.deptno(+) = d.deptno;


-- *** hr/hr 계정에서 test 
	-- cmd> sqlplus system/manager
	-- SQL> alter user hr idenified by hr account unlock;
	-- SQL> connect hr/hr

desc employees;	
desc jobs;

select count(*) from employees;
select count(*) from jobs;
select job_id from employees;
select distinct job_id from employees;
select job_id from jobs;


--16. 직원의 이름과 직책(job_title)을 출력(검색)
--	단, 사용되지 않는 직책이 있다면 그 직책이 정보도 검색에 포함
--     검색 정보 이름(first_name)과 job_title(직책) 

	-- 문제 풀이를 위한 table의 컬럼값들 확인해 보기
select employees.first_name, jobs.job_title
from employees, jobs
where jobs.job_id(+) = employees.job_id;

SELECT first_name, job_title
FROM employees e, jobs j
where e.job_id = j.job_id;

--outer join
--17. 직원들의 이름(first_name), 입사일, 부서명(department_name) 검색하기
-- 단, 부서가 없는 직원이 있다면 그 직원 정보도 검색에 포함시키기
select employees.first_name, employees.hire_date, departments.department_name
from employees, departments
where employees.department_id = departments.department_id(+);

/* 어딘가 잘못됐음 수정 요망
select count(departments.department_name)
from employees, departments
where departments.department_id(+) = employees.department_id;


select departments.department_name
from employees, departments
where employees.department_id(+) = departments.department_id and ;
*/

-- 미션 : 문제의 불명확스러움을 제시된 답안을 기점으로 이해하기 위한 sql문장 구성하기
-- 17. 직원들의 이름(first_name), 입사일, 부서명(department_name) 검색하기 

-- 단, 부서가 없는 직원이 있따면 그 직원 정보도 검색에 포함시키기
-- 경우의 수1 : 사원이 소속된 부서가 없는 경우
-- 경우의 수2 : 부서에 소속된 부서가 없는 경우






--문제 만들기. LOCATIONS와 DEPARTMENTS 테이블에서 location_id를
-- 이용하여 각 부서가 존재하는 나라의 코드를 출력하시오(나라의 코드는 중복된 것은 하나만 나올 수 있도록 한다.)
-- 단, 부서의 위치가 없는 곳은 제외한다.

SELECT distinct locations.country_id
FROM departments, locations
WHERE departments.location_id = locations.location_id(+);

select distinct l.country_id from departments d,locations l where d.location_id = l.location_id(+);
