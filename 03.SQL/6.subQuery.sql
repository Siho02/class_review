-- 6.subQuery.sql
-- select문 내에 포함된 또 다른 select문 작성 방법
-- 참고 : join 또는 subquery로 동일한 결과값 검색

-- 문법 : 비교 연산자(대소비교, 동등비교) 오른쪽에 () 안에 select문 작성 
--	   : create 및 insert 에도 사용 가능
--	   : tip - 개발시 다수의 test 위해서 원본이 아닌 복사본 table활용 권장 
-- 실행순서 : sub query가 main 쿼리 이전에 실행
-- tip : purge recyclebin
	-- 오라클 db가 자동 생성해주는 테이블, 사용자 관점에선 쓰레기
	-- 퍼지 리싸이클빈을 통해 정리 한번씩 해줌

/*
SNS 추천 알고리즘? = 추천 [클릭, 취향, 시청도, 방문 시간 및 기록 ... 탐색 ...]
	table 구조 설계
		개인정보 + 취향 + 고객 탐색 정보
		기본적으로 table 수
			경우의수1 : 하나의 table에 모두 지정
				-join 속도는 느리니 하나의 테이블에 설계
			경우의수2 :  고객 정보 테이블 / 클라이언트가 액션에 취하는 정보만 저장하는 취향 테이블 / 고객 탐색 테이블

		선호 방식은 2번 
			join / subquery 필수
			실행 속도 
*/


--1. SMITH라는 직원 부서명 검색
select dname
from emp, dept
where ename = 'SMITH' and emp.deptno = dept.deptno;

--sub query
--SMITH 사원의 부서 번호 검색 후에 해당 번호와 일치하는 부서테이블의 부서 번호와 일차하는 부서명 검색
select dname
from dept
where dept = (
	select deptno from emp where ename = 'SMITH'
)


--2. SMITH와 동일한 직급(job)을 가진 사원들의 모든 정보 검색(SMITH 포함)
select *
from emp
where job = (
	select job from emp where ename = 'SMITH'
) and ename != 'SMITH'


--3. SMITH와 급여가 동일하거나 더 많은(>=) 사원명과 급여 검색
-- SMITH 제외해서 검색하기
select ename, sal
from emp
where sal >= (
	select sal from emp where ename = 'SMITH'
) and ename != 'SMITH';


--4. DALLAS에 근무하는 사원의 이름, 부서 번호 검색
select ename, deptno
from emp
where deptno = (
	select deptno from dept where loc = 'DALLAS'
);



--5. 평균 급여보다 더 많이 받는(>) 사원만 검색
select *
from emp
where sal >= (
	select avg(sal) from emp 
);


-- 1~5번까지는 sub query의 결과가 단일



-- 다중행 서브 쿼리(sub query의 결과값이 하나 이상)
-- 6.급여가 3000이상 사원이 소속된 부서에 속한  사원이름, 급여 검색
	--급여가 3000이상 사원의 부서 번호
	-- in
select sal, deptno
from emp 
where dept in (
	select deptno from emp where sal >= 3000
);

--7. in 연산자를 이용하여 부서별로 가장 급여를 많이 받는 사원의 정보(사번, 사원명, 급여, 부서번호) 검색
select deptno, max(sal) from emp group by deptno;

select empno, ename, sal, deptno
from emp
where (deptno, sal) in (
	select deptno, max(sal)
	from emp
	group by deptno
);
	
--8. 직급(job)이 MANAGER인 사람이 속한 부서의 부서 번호와 부서명(dname)과 지역검색(loc)
select dname, loc, deptno
from dept
where deptno in (
	select deptno
	from emp
	where job = 'MANAGER'
);

