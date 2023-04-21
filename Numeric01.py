import numpy as np
import matplotlib.pyplot as plt


#Weekly tasks

array1 = [1, 2, 3, 4]

vektor1 = np.array(array1)

print("1 x 4 Zeilen Vektor: ", vektor1)

array2 = [[1],
          [2],
          [3],
          [4],
          [5]]

vektor2 = np.array(array2)

print("5 x 1 Zeilen Vektor: ", vektor2)

matrix_3d = np.zeros((3,3))
print(matrix_3d)

#Zum zeigen einfach eine Nullmatrix erstellt
matrix = np.zeros((4,3))
print(matrix[1])

matrix = np.zeros((4,4))
print(matrix[2])

matrix = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
print(matrix.transpose())

matrix1 = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
matrix2 = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
print(matrix1 * matrix2)

#Multiplikation
print(np.matmul(matrix1, matrix2))


matrix1 = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
matrix2 = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))
matrix3_c = np.c_[matrix2, matrix1]
matrix4_r = np.r_[matrix3, matrix3]
print("Spalten hinzugefügt: ")
print(matrix3_c)
print("_________________________________________________")
print("Zeilen hinzugefügt: ")
print(matrix4_r)

matrix1 = np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89]))

#Zeigt die Dimension an
print(matrix1.shape)

matrix1 = np.array(([11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33],
                   [11, 22, 33, 43, 77, 89, 33]))

print(matrix1.reshape(14,4))

vektor1 = np.array([[1],
                    [2],
                    [3]])
vektor2 = np.array([[1],
                    [2],
                    [3]])
for i in range(9999):
    vektor1 = np.c_[vektor1, vektor2]

print(vektor1)

matrix1 = np.array(([-11, 22, 33, 43, 77, -89, 33],
                    [11, 22, 33, -43, 77, 89, 33],
                    [11, 22, 33, 43, 77, 89, 33],
                    [11, 22, 33, 43, 77, 89, 33],
                    [11, -22, 33, 43, -77, 89, 33],
                    [11, 22, 33, 43, 77, 89, 33],
                    [11, 22, 33, 43, 77, 89, 33],
                    [11, -22, 33, -43, 77, 89, 33]))

for i in range(8):
    for j in range(7):
        # Kleiner null sollen 0 gesetzt werden
        if (matrix1[i][j] < 0):
            matrix1[i][j] = 0

print(matrix1)

matrix1 = np.array([[0]])
j = 0

for i in range(100):
    # Erster index soll 7 sein!
    if (i == 0):
        if (matrix1[0] == 0):
            matrix1[0] = 7

    if (i > 0):
        # Jetzt wird bis zu 7. Zahl gezählt
        j += 1
        if (j == 7):
            # Sobald 7 erreicht ist fangen wir an von vorne zu zählen
            j = 0;
            if (i != 7):
                matrix1 = np.r_[matrix1, [[i]]]

print(matrix1)

matrix1 = np.array([[0]])
j = 0

for i in range(100):
    matrix1 = np.r_[matrix1, [[i]]]

for i in range(100):
    if (i == 0):
        matrix1[i][0] = i
    # Modulo 2 = 1 weil bei null beginnend
    if (i % 2 == 1):
        matrix1[i][0] = 0

print(matrix1)

matrix1 = np.array([[0]])

for i in range(100):
    if (i != 0):
        matrix1 = np.r_[matrix1, [[i]]]

j = 1;
for i in range(101):
    if (j == 0):
        matrix1[j][0] = j
    if (i % 2 == 0 and i != 0):
        matrix1 = np.delete(matrix1, j, axis=0)
        j += 1

print(matrix1)

print(np.identity(3))

matrix1 = np.array(([-11, 22, 33, 43, 77],
                    [11, 22, 33, -43, 77],
                    [11, 22, 33, 43, 77],
                    [11, 22, 33, 43, 77],
                    [11, -22, 33, 43, -77]))
for i in range(5):
    print(matrix1[i][i])

np.diag(matrix1)

matrix1 =np.random.rand(1000,3)
matrix2 =np.random.rand(1000,3)
sk = [0,0,0]


startZeit = datetime.datetime.now()

print(matrix1)
for i in range(3):
    for j in range(1000):
        sk[i] += matrix1[j][i]*matrix2[j][i]

endZeit = datetime.datetime.now()
print(sk)
#Dauer Insgesamt
print("Zeit: ", endZeit - startZeit)

matrix1 =np.random.randint(100, size=(1000,4))
matrixInversen = np.zeros([1000, 4], dtype=float)
determinanten = np.empty(1000, dtype=int)
invertierbar = np.empty(1000, dtype=bool)

for j in range(1000):
        determinanten[j] = matrix1[j][0] * matrix1[j][3] -  matrix1[j][1] * matrix1[j][2]
        if(determinanten[j] != 0):
            invertierbar[j] = True;
        else:
            invertierbar[j] = False;

print(matrix1)
print("!!__________________________________________________!!")
print(determinanten)


for j in range(1000):
    if(invertierbar[j] == True):
        matrixInversen[j][0] =   float(matrix1[j][3]/determinanten[j])
        print(matrix1[j][3], ":" , determinanten[j])
        matrixInversen[j][1] =    float(-matrix1[j][1]/determinanten[j])
        print(-matrix1[j][1], ":" ,  float(determinanten[j]))
        matrixInversen[j][2] =    float(-matrix1[j][2]/determinanten[j])
        print(-matrix1[j][2], ":" , determinanten[j])
        matrixInversen[j][3] =   matrix1[j][0]/determinanten[j]
        print(matrix1[j][0], ":" ,  float(determinanten[j]))

#Erste Matrix zum Überprüfen. Klappt soweit nur durch float nicht ganz genau!
print(matrix1[0])
print(matrixInversen[0])


XWerte = np.random.randint(10, size=(20))
YWerte = np.empty(20, dtype=float)
for i in range(20):
    YWerte[i] = XWerte[i]*2
print(XWerte)
print(YWerte)
RauschWerte = np.random.normal(0, 1, YWerte.shape)
for i in range(20):
    YWerte[i] += RauschWerte[i]
print(YWerte)

z = np.polyfit(XWerte, YWerte, 1)
xp = np.linspace(0, 20, 20)
_ = plt.plot(XWerte, YWerte, '.', xp, '-', xp, '--')
plt.ylim(0, 20)
plt.show()
print(YWerte)

XWerte2 = np.random.randint(-10, 10, size=(20))
YWerte2 = np.empty(20, dtype=float)
for i in range(20):
    YWerte2[i] = XWerte2[i]^2
print(XWerte2)
print(YWerte2)
RauschWerte2 = np.random.normal(0, 20, YWerte2.shape)
for i in range(20):
    YWerte2[i] += RauschWerte2[i]
print(YWerte2)

z2 = np.polyfit(XWerte, YWerte, 2)
p1 = np.poly1d(z2)
xp2 = np.linspace(0, 200, 20)
_ = plt.plot(XWerte2, YWerte2, '.', xp2, p1(xp) , '-')
plt.ylim(0, 200)
plt.show()
print(YWerte2)
