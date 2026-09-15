"""Published numerical references, transcribed from arXiv:2011.14473v2.

R² in these tables denotes the paper's mean squared Pearson correlation.
Tolerances are half a unit in the last printed digit, plus numerical slack.
"""
import numpy as np
from . import load_archive, models
from .reference import evaluate

SOURCE = 'https://arxiv.org/html/2011.14473v2'
# Each tuple is (r², MAE, MSE), as printed in Table 3; strings retain rounding.
FORWARD = [
    [('1.00','5.50e-4','3.92e-7'), None, ('1.00','1.79e-5','4.27e-10'), None],
    [('1.00','2.51e-3','1.12e-5'), ('1.00','2.97e-3','1.84e-5'), ('1.00','8.93e-3','2.49e-4'), ('1.00','1.73e-2','1.32e-3')],
    [('1.00','3.30e-3','2.09e-5'), ('0.999','1.26e-2','4.70e-4'), ('1.00','2.78e-3','1.68e-5'), ('1.00','7.38e-3','1.42e-4')],
    [('1.00','2.18e-3','9.87e-6'), ('1.00','7.13e-3','4.00e-4'), ('1.00','6.89e-4','8.45e-7'), ('1.00','3.33e-3','3.58e-5')],
]
# Table 4, one row per (model, IC, scenario): derivatives then surface states.
INVERSE = [
    ['1.00','9.59e-3','2.99e-4','0.972','3.17e-2','1.40e-3'],
    ['0.944','0.131','0.325','0.999','1.09e-3','2.30e-6'],
    ['0.941','7.62e-2','1.90e-2','0.863','1.56e-2','4.79e-4'],
    ['1.00','7.59e-3','1.69e-4','0.998','3.48e-2','1.69e-3'],
    ['0.998','7.82e-2','4.47e-2','1.00','1.26e-3','2.23e-6'],
    ['1.00','4.47e-2','3.37e-3','0.948','1.79e-2','4.15e-4'],
    ['1.00','1.91e-2','5.92e-4','0.965','0.118','2.32e-2'],
    ['0.999','2.63e-2','3.50e-3','1.00','6.39e-4','6.21e-7'],
    ['0.983','5.96e-2','5.93e-3','0.895','2.67e-2','1.29e-3'],
    ['0.996','1.04e-2','1.54e-4','0.940','0.151','3.68e-2'],
    ['1.00','1.13e-2','4.60e-4','1.00','6.64e-4','6.96e-7'],
    ['0.988','4.08e-2','3.70e-3','0.988','3.24e-2','2.05e-3'],
    ['0.975','1.87e-2','8.11e-4','0.374','0.146','3.84e-2'],
    ['1.00','5.53e-3','8.53e-5','1.00','2.66e-3','1.28e-5'],
    ['1.00','1.28e-2','3.24e-4','0.996','1.47e-2','3.89e-4'],
    ['0.736','1.81e-2','9.85e-4','0.236','0.181','5.76e-2'],
    ['1.00','9.30e-3','2.51e-4','1.00','2.68e-3','1.30e-5'],
    ['0.998','2.11e-2','1.44e-3','0.989','1.06e-2','1.45e-4'],
]
# Table 5: MAE by reaction group and the total Pearson correlation.
PARAMETERS = {
    (0,'inv'): {'g':'9.90e-4','rho':'1.00'}, (0,'invvwn'): {'g':'2.29e-4','rho':'1.00'},
    (1,'inv'): {'d':'0.161','a':'9.76e-2','rho':'0.991'},
    (1,'invsc'): {'d':'5.95e-2','a':'1.16e-2','rho':'0.999'},
    (1,'invvwn'): {'d':'0.150','a':'8.11e-2','rho':'0.999'},
    (2,'inv'): {'d':'0.520','c':'1.80e1','rho':'-0.523'},
    (2,'invsc'): {'d':'1.70e-2','c':'7.32e-2','rho':'1.00'},
    (2,'invvwn'): {'d':'0.236','c':'0.420','rho':'0.996'},
    (3,'inv'): {'d':'5.06','c':'1.02e1','s':'1.90','rho':'0.205'},
    (3,'invsc'): {'d':'4.10e-2','c':'0.116','s':'6.61e-2','rho':'0.999'},
    (3,'invvwn'): {'d':'0.110','c':'0.114','s':'9.62e-2','rho':'0.997'},
}


def comparison(actual, printed, **context):
    from decimal import Decimal
    tolerance = float(Decimal(10) ** Decimal(printed).as_tuple().exponent) / 2
    # Small integration / floating-point allowance, not an accuracy tolerance.
    tolerance += abs(float(printed)) * 1e-7 + 1e-12
    return {**context, 'published': printed, 'calculated': actual, 'absolute_tolerance': tolerance,
            'matches': actual is not None and abs(actual - float(printed)) <= tolerance}


def verify():
    rows = []
    metrics = ('r2', 'mae', 'mse')
    cache = {}
    def result(index, scenario):
        key = f'trainer_{scenario}_{index}_alpha13'
        if key not in cache:
            cache[key] = evaluate(key)
        return cache[key]
    for index, references in enumerate(FORWARD):
        r = result(index, 'fwd')
        for offset, expected in enumerate(references):
            if expected is None:
                continue
            group = 'surface_derivatives' if offset % 2 else 'bulk_derivatives'
            for metric, printed in zip(metrics, expected):
                rows.append(comparison(r['datasets'][offset//2]['metrics'][group][metric], printed,
                                       table=3, experiment=r['id'], ic=offset//2+1, group=group, metric=metric, checkpoint=r['selected_state']))
    for index in range(1,4):
        for ic in range(2):
            for scenario_index, scenario in enumerate(('inv','invsc','invvwn')):
                expected = INVERSE[(index-1)*6 + ic*3 + scenario_index]
                r = result(index, scenario)
                for group, values in zip(('surface_derivatives','surface_states'), (expected[:3],expected[3:])):
                    for metric, printed in zip(metrics, values):
                        rows.append(comparison(r['datasets'][ic]['metrics'][group][metric], printed,
                                               table=4, experiment=r['id'], ic=ic+1, group=group, metric=metric, checkpoint=r['selected_state']))
    for (index, scenario), expected in PARAMETERS.items():
        identifier = f'trainer_{scenario}_{index}_alpha13'
        archive = load_archive(identifier)
        # The published table matches the penultimate noiseless states and final
        # noisy states. This correspondence is inferred from the saved numbers;
        # the checked-in plotting script has a one-state lag in its mpars list.
        state = None if scenario == 'invvwn' else len(archive['iter_data']) - 2
        params = archive['params'] if state is None else archive['iter_data'][state][1][0]
        logs, truth = np.asarray(params[1][0]), np.log(models()[str(index)]['kijnpars'])
        groups = {'g':[0,1]} if index == 0 else {'d':list(range(6)), ('a' if index==1 else 'c'):list(range(6,len(logs)))}
        if index == 3:
            groups = {'d':list(range(6)), 'c':[6,7,8,9,12,13], 's':[10,11]}
        for group, printed in expected.items():
            actual = float(np.corrcoef(logs, truth)[0,1]) if group == 'rho' else float(np.mean(np.abs(logs[groups[group]]-truth[groups[group]])))
            rows.append(comparison(actual, printed, table=5, experiment=identifier, group=group,
                                   metric='correlation' if group=='rho' else 'log_rate_mae', checkpoint=state))
    failed = [row for row in rows if not row['matches']]
    return {'paper': '2011.14473v2', 'source': SOURCE, 'scope':'archived parameters against Tables 3, 4 and 5; not a fresh-training equivalence test',
            'status': 'matches' if not failed else 'differences_found', 'checks':len(rows), 'matched':len(rows)-len(failed),
            'differences':failed, 'comparisons': rows,
            'checkpoint_note':'Table 4 follows the original noisy-checkpoint heuristic. Table 5 checkpoint correspondence is inferred from the printed values: penultimate for noiseless cases, final for noisy cases.'}
