import langchain


print(langchain.__version__)



var1= "Yash Jadon"


print(var1)


print()


n = int(input("Enter the value of n: "))

for i in range(n):
    for j in range(i+1):
        print("*",end="")
    print()   



n = 3

for i in range(n):
    # spaces
    for j in range(n - i - 1):
        print(" ", end="")

    # stars
    for j in range(2 * i + 1):
        print("*", end="")

    print()


print()


for i in range(3):
    for j in range(3):
        print((i*3)+j,end="")
    
    print()