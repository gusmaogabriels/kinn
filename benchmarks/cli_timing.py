"""Measure fresh CLI processes and their synchronized training kernels."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('build/cli-timing.json'))
args = parser.parse_args()
results = []
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for method in ('kinn', 'rkinn'):
        for mode in ('forward', 'inverse'):
            problem, output = root/'problem.json', root/'result.json'
            subprocess.run([sys.executable, '-m', 'kinn', 'example', '--method', method, '--mode', mode, '--output', str(problem)], check=True)
            started = perf_counter()
            process = subprocess.run([sys.executable, '-m', 'kinn', 'run', str(problem), '--output', str(output)], check=False)
            process_wall = perf_counter() - started
            result = json.loads(output.read_text())
            assert process.returncode == 0 and result['status'] == 'converged', result
            timing = result['timing']
            results.append({'method': method, 'mode': mode, 'fresh_process_wall_seconds': process_wall,
                            'training': timing, 'median_warm_epoch_seconds': statistics.median(timing['warm_epoch_seconds']) if timing['warm_epoch_seconds'] else None,
                            'rate_constants': result['rate_constants'], 'final_residuals': result['history'][-1]})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'platform': platform.platform(), 'python': sys.version,
    'scope': 'Fresh CLI process includes imports and compilation. Warm epochs reuse the compiled training kernel within one solve. Timings are informational, not cross-machine performance thresholds.',
    'results': results}, indent=2)+'\n')
print(args.output)
