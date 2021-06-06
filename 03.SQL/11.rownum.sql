--11.rownum.sql

-- *** rownum
-- oracle 자체적으로 제공하는 컬럼
-- table 당 무조건 자동 생성
-- 검색시 검색된 데이터 순서대로 rownum값 자동 반영(1부터 시작)

-- *** 인라인 뷰
	-- 검색시 빈번히 활용되는 스펙
	-- 다수의 글들이 있는 게시판에 필수로 사용(paging 처리)
	-- 서브쿼리의 일종으로 from절에 위치하여 테이블처럼 사용
	-- 원리 : sql문 내부에 view를 정의하고 이를 테이블처럼 사용 

select rownum, empno from emp;
select rownum, deptno from dept;

-- 코드만으로 rownum 탐색?
-- 검색시 검색된 데이터 순서대로 rownum값 자동 반영(1부터 시작)
-- 실행 순서 : from >> where >> select
-- 암시사항 : rownum은 검색시에 검색된 결과에 자동 index를 부여 
	-- 1부터 활용해야함
	-- rownum > 4 인 경우 5부터 시작점으로 내부적으로 간주 
	-- 	    >> 문법 오류가 아닌 논리적 오류(1부터 시작해야한다는 활용x)
	-- from절 select절 where절 순으로 실행되는 것은 변한 없으나 
		-- where절에서 사용시 1부터 유효한 경우에 한해서만 정상 인지
		-- rownum은 오라클 자체적인 키워드 즉, 이미 존재하는 기능
-- 모든 언어가 키워드는 적절한 위치에 있으면 문법 오류는 발생 x 

/* 인라인 뷰
 */

-- ---------------------
/*
inline view 방식
	from절에 select문장으로 검색된 데이터가 반영되는 구조를 inline
	임시로 생성된 table로 간주 즉 물리적으로 존재하지는 table로 간주
	논리적인 table 즉 view

select 검색 컬럼
from 존재하는table 또는 검색된 데이터(임시table)
*/
select rownum, deptno from dept;
select rownum, deptno from emp;

select rownum, deptno 
from (select rownum, deptno 
	 from dept 
	 where rownum < 4);


select rownum, deptno 
from (select rownum, deptno 
	 from dept 
	 where rownum < 4);



-- 1. ? dept의 deptno를 내림차순(desc)으로 검색, rownum
select rownum, deptno from dept order by deptno desc;
select rownum, deptno from dept order by deptno asc; --(오름차순 정리)

-- 2. ? deptno의 값이 오름차순으로 정렬해서 30번 까지만 검색, rownum 포함해서 검색

select rownum, deptno 
from dept 
where rownum < 4
order by deptno desc;

-- 3. ? deptno의 값이 오름차순으로 정렬해서 상위 3개의 데이터만 검색, rownum 포함해서 검색
select rownum, deptno 
from dept 
where rownum < 4
order by deptno asc;


-- 4.  인라인 뷰를 사용하여 급여를 많이 받는 순서대로 3명만 이름과 급여 검색 
select enmae, sal from emp order by sal desc;
select rownum, ename, sal from emp order by sal desc;
-- from >> select(rownum) >> order by sal ... : 따라서 어그러짐
/*
select rownum, ename, sal
from (select rownum, ename, sal
	 from emp 
	 order by sal desc);
*/

select rownum, sal
from (select sal 
	 from emp 
	 order by sal desc)
where rownum  < 4;


--5. emp table의 deptno의 값이 오름차순으로 정렬된 상태로 상위 3개 데이터 검색

select rownum, ename, deptno 
from (select ename, deptno
	from emp 
	order by deptno asc)
where rownum < 4;