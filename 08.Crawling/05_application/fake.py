from faker import Faker

fake = Faker(['ko-KR'])
print(fake.name())
print(fake.email())
print(fake.country())

print(fake.profile())


print('--------------------------')

# _ : 묵시적으로 문법은 필요에 의해 선언은 하지만 사용은 하지 않는 변수
for _ in range(10):
    print(fake.name())
