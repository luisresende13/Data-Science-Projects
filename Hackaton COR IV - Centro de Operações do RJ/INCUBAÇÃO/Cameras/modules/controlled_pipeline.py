import pandas as pd
from IPython.display import clear_output as co
from time import time

class Progress:
    
    def __init__(self, n, n_total=None):
        self.s = time(); self.n = n; self.n_total = n_total

    def report(self, i):
        # update statistics
        t = time() - self.s; left = self.n - i
        t_min = round(t / 60, 1)
        progress_prct = round(i / self.n * 100, 1)
        if i != 0:
            rate = t / i
            t_finish = left * rate
            t_finish_min = round(t_finish / 60, 1)
        if self.n_total is not None:
            i_total = self.n_total - self.n + i
            total_prct = round(i_total / self.n_total * 100, 1)
        # print report
        print(f'\n- PROGRESS: {i} / {self.n} ops · PROGRESS-PRCT: {progress_prct} %')
        if self.n_total is not None:
            print(f'\n- TOTAL: {i_total} / {self.n_total} ops · TOTAL-PRCT: {total_prct} %')
        if i != 0: print(f'\n- RUNNING: {t_min} min · EXPECT-FINISH: {t_finish_min} min · RATE: {round(1/rate, 4)} ops / s')
            
def df_query(data, query):
    df = data.copy()
    for key in query: df = df[df[key].isin(query[key] if type(query[key]) is list else [query[key]])]
    return df

class ControlledPipeline:
    
    def __init__(
        self, control_path, control_func, params_fields, query={},
        status_field='status', status_options={True: 'SUCCESS', False: 'FAILED'},
        error_flag='ERROR',
    ):
        self.control_path = control_path; self.control_func = control_func
        self.params_fields = params_fields; self.query = query
        self.status_field = status_field; self.status_options = status_options;
        self.error_flag = error_flag
        
    def run(self, report_each=10, save_each=100):

        blobs_control = pd.read_csv(self.control_path)
        control = df_query(blobs_control, self.query)

        n = len(control)
        n_total = len(blobs_control)
        progress = Progress(n, n_total)

        try:
            for i, (index, params) in enumerate(control.iterrows()):
                if report_each is not None and i % report_each == 0:
                    co(True); progress.report(i)
                if save_each is not None and i != 0 and i % save_each == 0:
                    blobs_control.to_csv(self.control_path, index=False) # save status control
                    print('\n* BLOBS CONTROL UPDATE SUCCESSFUL!')
                try:
                    status = self.control_func(**params[self.params_fields].to_dict())
                    status_name = self.status_options[int(status)]
                except Exception as err:  # does not catch KeyboardInterrupt
                    status = None; status_name = self.error_flag
                    print(f'\n* EXCEPTION: {err} · UPDATE-STATUS: {status_name}')
                blobs_control.loc[index, self.status_field] = status_name

        except KeyboardInterrupt as err:
            print('\nPROCESS ENDED')
            print(f'\n* EXCEPTION: {err}')

        progress.report(i)
        blobs_control.to_csv(self.control_path, index=False) # save status control
        print('\n* BLOBS CONTROL UPDATE SUCCESSFUL!')