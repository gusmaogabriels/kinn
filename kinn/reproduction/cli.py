"""CLI entry points for the paper archives and their numerical audit."""
import json
from pathlib import Path
import sys
from time import perf_counter

from . import catalog, load_archive, resource


def add_parser(subparsers):
    parser = subparsers.add_parser('reproduce', help='Evaluate paper archives, verify published tables, or rerun a training schedule locally.')
    sub = parser.add_subparsers(dest='reproduction_command', required=True)
    sub.add_parser('list', help='List experiments and coverage for both papers.')
    verify = sub.add_parser('verify', help='Compare archived results with published Tables 3–5; exit 1 if any values differ.')
    verify.add_argument('--output')
    plan = sub.add_parser('plan', help='Print the architecture, data-generation settings and archived stage schedule.')
    plan.add_argument('experiment')
    plan.add_argument('--output')
    run = sub.add_parser('run', help='Re-evaluate an archived experiment or all 21 archives; export arrays, parameters and figures.')
    run.add_argument('experiment', help='An experiment ID from list, or all.')
    run.add_argument('--state', default='paper', help='paper (original plot selection), final, or a zero-based saved checkpoint index.')
    train = sub.add_parser('train', help='Train afresh with an archived architecture and stage schedule; full runs can take hours.')
    train.add_argument('experiment')
    train.add_argument('--epochs-per-stage', type=int)
    train.add_argument('--steps-per-epoch', type=int)
    train.add_argument('--max-stages', type=int)
    for child in (run, train):
        child.add_argument('--output', required=True, help='New or empty local output directory.')
        child.add_argument('--plots', action='store_true', help='Export SVG/PDF plots; requires kinn[reproduce].')


def run(args):
    from ..cli import _emit
    command = args.reproduction_command
    if command == 'list':
        _emit({**catalog(), 'coverage': json.loads(resource('coverage.json').read_text())})
        return 0
    if command == 'verify':
        from .published import verify
        from .output import environment
        report = verify()
        report['environment'] = environment()
        _emit(report, args.output)
        return 0 if report['status']=='matches' else 1
    if command == 'plan':
        from .training import plan
        _emit(plan(args.experiment), args.output)
        return 0
    from .output import export, environment, save_tree, require_plots
    if args.plots:
        require_plots()
    target = Path(args.output)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Choose a new or empty output directory to preserve previous results')
    if command == 'train':
        from .training import train, plan
        # Validate the recipe and overrides before creating output files.
        plan(args.experiment)
        for name in ('epochs_per_stage','steps_per_epoch','max_stages'):
            value = getattr(args,name)
            if value is not None and value < 1:
                raise ValueError(f'{name} must be a positive integer')
        target.mkdir(parents=True,exist_ok=True)
        def checkpoint(archive, recipe):
            directory = target / 'checkpoints' / f"stage-{len(archive['iter_data']):03d}"
            directory.mkdir(parents=True,exist_ok=True)
            save_tree(directory,'checkpoint',archive)
            _emit(recipe, directory/'recipe.json')
        started = perf_counter()
        from ..numerics import NumericalError
        try:
            result, archive = train(args.experiment, epochs_per_stage=args.epochs_per_stage,
                                    steps_per_epoch=args.steps_per_epoch,max_stages=args.max_stages,
                                    progress=lambda row: print(json.dumps(row),file=sys.stderr,flush=True),checkpoint=checkpoint)
        except NumericalError as error:
            report = {'status':'numerical_failure','error':str(error),'experiment':args.experiment,
                      'total_wall_seconds':perf_counter()-started}
            _emit(report,target/'failure.json')
            _emit(report)
            return 1
        result['environment'] = environment()
        export(result,target,plots=args.plots)
        save_tree(target,'parameters',archive['params'])
        _emit({'status':result['status'],'output':str(target),'total_wall_seconds':perf_counter()-started})
        return 0
    from .reference import evaluate
    from . import record
    state = args.state
    if state not in ('paper','final'):
        try:
            state = int(state)
        except ValueError as error:
            raise ValueError('state must be paper, final, or a nonnegative checkpoint index') from error
    ids = [row['id'] for row in catalog()['records']] if args.experiment=='all' else [record(args.experiment)['id']]
    if isinstance(state,int) and any(not 0 <= state < record(identifier)['saved_states'] for identifier in ids):
        raise ValueError('Checkpoint index is out of range for one or more selected experiments')
    target.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    manifest = {'schema_version':1,'origin':'archived_parameters','execution':'local', 'results':[],
                'coverage': json.loads(resource('coverage.json').read_text())}
    for identifier in ids:
        result = evaluate(identifier,state=state)
        folder = target/identifier if args.experiment=='all' else target
        export(result,folder,plots=args.plots)
        archive = load_archive(identifier)
        selected = result['selected_state']
        params = archive['params'] if selected is None else archive['iter_data'][selected][1][0]
        save_tree(folder,'parameters',params)
        # Include all original stage weights for parameter-history retrieval.
        save_tree(folder,'checkpoints',archive)
        manifest['results'].append({'experiment':identifier,'selected_state':selected,'result':str((folder/'result.json').relative_to(target))})
    manifest.update(environment=environment(), total_wall_seconds=perf_counter()-started)
    if args.experiment == 'all':
        from .published import verify
        verification = verify()
        _emit(verification,target/'verification.json')
        manifest['published_verification'] = {key:verification[key] for key in ('status','checks','matched','differences')}
        manifest['total_wall_seconds'] = perf_counter()-started
    _emit(manifest,target/'manifest.json')
    _emit({'status':'archives_evaluated','experiments':len(ids),'output':str(target),
           'total_wall_seconds':manifest['total_wall_seconds'], 'full_paper_reproduction':False})
    return 0
