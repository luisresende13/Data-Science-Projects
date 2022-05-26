
patm = 101325 #Pa
gama = 9807 #N
p = 1000 #N/m**3
Mh2o = 0.034 #kg/mol
g = 9.81 #m/s**2
R = 0.082 * patm
control = int(input('control:'))
a = ('''A1 = float(input('A1:'))
h = float(input('h:'))
r1 = float(input('r1:'))
r2 = float(input('r2:'))
r3 = float(input('r3:'))
F1 = float(input('F1:'))
F2 = float(input('F2:'))
V1 = float(input('V1:'))
V2 = float(input('V2:'))
V3 = float(input('V3:'))
p1 = float(input('p1:'))
p2 = float(input('p2:'))
0

if A1!='': A1 = math.pi*r1**2
if A2!='': A2 = math.pi*r2**2
if A3!='': A3 = math.pi*r3**2
''')
control = int(input('control:'))

if control == 0:

    A1 = float(input('A:')) * 10**(-4)
    patm = float(input('patm:'))
    T = float(input('T:'))
    T = T + 273 # Kelvin
    h1 = float(input('h1:'))
    h2 = float(input('h2:'))
    p = float(input('p:'))
    gama = g*p
    
    V = A1 * (h1-h2)
    R = 0.082 * 101325
    P1 = patm + h2*gama
    n = P1*V/(R*T)
    m = n*Mh2o
    print('massa:', m)

elif control == 1:

    A1 = (F2/A2 + p*g*h) / F1
    dF = F1*A2 / A1 - F2
    print('acrescimo de força:', dF)
elif control == 2:

    Q = A3*V3
    V2 = Q / A2
    V1 = Q/A1

    p2 = patm + (p*V3**2)/2 - (p*V2**2)/2
    p1 = patm + (p*V3**2)/2 - (p*V1**2)/2

    h = p2/gama
    print('Pressão 1:', p1)
    print('h:', h)
elif control == 3:

    #V2 = A1*V1/A2
    #V1 = ((p * V2**2 + z2 - z1)*2/p)**0.5
    
    V1 = ((- z1 + z2) / (1/2 - ((p/2)*(A1/A2)**2)))**0.5
    print('Velocidade cheio:', V1)

