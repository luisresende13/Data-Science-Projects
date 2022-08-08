import math


control = input('''0: Classificação de rugosidade\n
                1: Alturas e pressões\n
                2: Determinar pressão em A\n
                Insira 0, 1 ou 2 para continuar: ''')

if control=='0':

    Re = float(input('Insira Re:'))
    rug_rel = float(input('Insira a rugosidade relativa:'))

    f = 0.25/math.log10(rug_rel/3.7 + 5.74/(Re**0.9))**2

    param_class = rug_rel * Re * (f/8) **(1/2)

    print('f:', f)
    print('parâmetro de classificação da rugosidade (e):', param_class)

    if param_class < 5: msg = 'hidraulicamente liso'
    elif param_class <= 70: msg = 'rugosidade transicional'
    else :msg = 'totalmente rugoso'
    print('classificação:', msg)

elif control=='1':
    print('Dados do Problema:'); gama = 10e3
    D, L, za, pa, zb, pb, Q = [float(input(text)) for text in ['Diâmetro da tubulação:',
                                                            'Comprimento:',
                                                            'Altura ponto A:',
                                                            'Pressão ponto A:',
                                                            'Altura ponto B:',
                                                            'Pressão ponto B',
                                                            'Vazão:'
                                                            ]]

    dH =  pb/gama + zb - pa/gama - za
    
    
    
    print(); print('Resultado:'); print()
    if dH < 0: print('Sentido do fluxo: A>B'); dH = dH*(-1)
    else: print('Sentido do fluxo: B>A')
    
    f = dH * (D**5) / (L * 0.0826 * Q**2)
    Va = (dH*D*9.8/(4*L))**(1/2)

    print('Perda de carga (hp):', dH)
    print('Fator de atrito (f):', f)
    print('Velocidade de atrito (u*):', Va)

elif control=='2':

    D, rug, L, Q, v = [float(input(text)) for text in ['Diâmetro da tubulação (D):',
                                                    'Rugosidade (e):',
                                                    'Comprimento (L):',
                                                    'Vazão (Q):',
                                                    'Viscosidade cinemática (v):'
                                                       ]]

    D = D * 0.0254
    A = math.pi*(D**2)/4 
    V = Q/A
    Re = V * D / v
    rug_rel = rug/D

    f = 0.25/math.log10(rug_rel/3.7 + 5.74/(Re**0.9))**2
    hp = (L*f*V**2)/(D*2*9.8)

    [print(texto, valor) for texto, valor in zip(['Rugosidade relativa:',
                                                  'Velocidade:',
                                                  'Reynolds:',
                                                  'fator de atrito:',
                                                  'perda de carga:'],
                                                 [rug_rel, V, Re, f, hp])]

    

    
    
