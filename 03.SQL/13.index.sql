--13.index.sql
/*
1. db의 빠른 검색을 위한 색인 기능의 index 학습
	- primary key는 기본적으로 자동 index로 설정됨 
	
	- DB 자체적으로 빠른 검색 기능 부여
		이 빠른 검색 기능 - index
	- 어설프게 사용자 정의 index 설정시 오히려 검색 속도 다운
	- 데이터 셋의 15% 이상의 데이터들이 잦은 변경이 이뤄질 경우 index 설정 비추

2. 실행 속도 체크를 위한 옵션 명령어
	set timing on
	set timing off

3. sql명령문의 검색 처리 속도 향상을 위한 oracle db 자체의 객체


4. 주의사항
	- index가 반영된 컬럼 데이터가 수시로 변경되는 데이터라면 index 적용은 오히려 부작용
*/

--1. index용 검색 속도 확일을 위한 table 생성
drop table emp01;
create table emp01 as select * from emp; 	-- index 적용해서 실행 속도 확인을 위해 수십 수백 번씩 수행했다 가정


--2. 테스트를 위한 데이터값을 복사 붙여넣기
insert into emp01 select * from emp01;
select count(*) from emp01;

select * from emp01 where ename='SMITH';


--3. emp01 table에 없던 index(색인) 기능을 적용하는 문법
create index idx_emp01_empno on emp01(empno)



--4. SMITH 사번 검색 시간 체크  
select * from emp01 where empno=7369;
select * from emp01 where ename = 'SMITH';

--5. index 삭제 명령어
drop index idx_emp01_empno;

drop table emp01;
 