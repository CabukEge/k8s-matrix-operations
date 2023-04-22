import numpy as np

A = np.matrix('1 -1 1; 1 0 0; 1 1 1; 1 2 4')
b = np.array([[1], [0], [2], [5]])


AA = np.matmul(A.transpose(),A)
Ab = np.matmul(A.transpose(),b)
AAInverse = np.linalg.inv(AA)
Ergebnis = np.matmul(AAInverse,Ab)

print("A =",A)
print("________________")
print("b =",b)
print("________________")
print("A*A =",AA)
print("________________")
print("A*b =",Ab)
print("________________")
print("(A*A)^-1 =",AAInverse)
print("________________")
print("Test: ((A*A)^-1) * A =")
print(np.matmul(AA,AAInverse))
print("________________")
print(Ergebnis)
print("________________")


nGridCell=3
XNew=np.linspace(0.0,1.0,nGridCell)
YNew=np.linspace(0.0,1.0,nGridCell)
XNew,YNew=np.meshgrid(XNew,YNew)
print(XNew.shape)

nPoint=3
X=np.random.rand(nPoint)
Y=np.random.rand(nPoint)
Z=np.random.rand(nPoint)
print(X)