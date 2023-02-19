import pandas as pd, requests

class cameras:

    api_root = 'https://bolsao-api-j2fefywbia-rj.a.run.app'

    def __init__(self, src, monitor, logic='any', merge=False):
        self.src = src
        self.monitor = monitor
        self.logic = logic
        self.merge = merge
        self.static = self.read_df('cameras')
        self.update()

    def read_df(self, src):
        return pd.DataFrame(requests.get(f'{self.api_root}/{src}').json())

    def in_alert_msk(self):
        if self.logic == 'any':
            return self.state[self.monitor].any(axis=1)
        elif self.logic == 'all':
            return self.state[self.monitor].all(axis=1)

    def update(self, report=False):
        self.state = self.read_df(self.src)
        if self.merge: self.state =  pd.merge(self.static, self.state, how='inner', on='cluster_id')
        self.in_alert = self.state['Codigo'][self.in_alert_msk()].tolist()
        self.normal = list(set(self.state['Codigo']).difference(self.in_alert))
        if report: self.report()
        
    def report(self): display(pd.DataFrame(
        [[len(self.in_alert), len(self.normal), len(self.state)]],
        columns=['In alert', 'Normality', 'Total'], index=[self.src]
    ))