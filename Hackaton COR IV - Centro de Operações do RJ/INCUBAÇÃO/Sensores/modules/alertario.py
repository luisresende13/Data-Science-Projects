''' EXAMPLE USAGE:
    from modules.alertario import alertario
    alerts = alertario.alert()
    print(alerts.head())
'''

import requests, pandas as pd

url = r'http://websempre.rio.rj.gov.br/json/chuvas'

class alertario:
    
    def alert():
        data = requests.get(url).json()['objects']
        df = pd.DataFrame([{**obj, **obj['data']} for obj in data])
        df['id_estacao'] = df['name'].map(alerta_station_name_id_map).astype(str)
        return df.drop('data', axis=1).rename(columns=alerta_feature_name_map)
    
alerta_feature_name_map = {
    'm05': 'acumulado_chuva_5_min',
    'm15': 'acumulado_chuva_15_min',
    'h01': 'acumulado_chuva_1_h',
    'h02': 'acumulado_chuva_2_h',
    'h03': 'acumulado_chuva_3_h',
    'h04': 'acumulado_chuva_4_h',
    'h24': 'acumulado_chuva_24_h',
    'h96': 'acumulado_chuva_96_h',
    'mes': 'acumulado_chuva_1_mes',
}

alerta_station_name_id_map = {
    'Tijuca': 4,
    'Guaratiba': 20,
    'Santa Cruz': 22,
    'Sepetiba': 27,
    'Rocinha': 3,
    'Madureira': 10,
    'Alto da Boa Vista': 28,
    'Laranjeiras': 31,
    'Piedade': 13,
    'Campo Grande': 26,
    'Ilha do Governador': 8,
    'Penha': 9,
    'Vidigal': 1,
    'Tijuca/Muda': 33,
    'Urca': 2,
    'Santa Teresa': 5,
    'Copacabana': 6,
    'Anchieta': 24,
    'Grota Funda': 25,
    'Bangu': 12,    
    'Av. Brasil/Mendanha': 29,
    'Barra/Barrinha': 17,
    'Barra/Riocentro': 19,
    'Est. Grajaú/Jacarepaguá': 21,
    'Grajaú': 7,
    'Grande Méier': 23,
    'Irajá': 11,
    'Jacarepaguá/Cidade de Deus': 18,
    'Jacarepaguá/Tanque': 14,
    'Jardim Botânico': 16,
    'Recreio dos Bandeirantes': 30,
    'Saúde': 15,
    'São Cristóvão': 32,
}