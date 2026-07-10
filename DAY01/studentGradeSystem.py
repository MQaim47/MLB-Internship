name=input("Enter Student Name:")
std_class=input("Enter Class:")

num_subjects=int(input("Enter Number of Subjects:"))
subjects={}
total=0

for i in range(num_subjects):
    subject=input(f"Enter Subject{i+1}:")
    while True: 
        marks=float(input(f"Enter marks for {subject}:"))
        if 0<= marks <=100:
            break
        else:
            print("Invalid Marks")
    subjects[subject]=marks
    total += marks
    print("\n")
    
average=total/num_subjects

if average>=90:
    grade="A"
    
elif average>=80:
    grade="B"
elif average>=70:
    grade="C"
elif average>=50:
    grade="D"
else:
    grade="F"
    
print("\n----- Result -----")
print("Name:", name)
print("Class:", std_class)
print("Average:", average)
print("Grade:", grade)