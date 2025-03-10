print("Ihr werdet mich niemals besiegen!")
eingabe = input("Gebe +,-,*,/ an: ")
a = float(input("Zahl 1: "))
b = float(input("Zahl 2: "))

if eingabe=="+":
    print(a+b)
elif eingabe == "-":
    print(a-b)
elif eingabe == "*":
    print(a*b)
elif eingabe == "/":
    print(round(a/b,2))
else:
    print("Falsche eingabe")