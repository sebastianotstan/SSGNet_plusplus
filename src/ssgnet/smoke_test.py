"""One-command offline CPU smoke test; writes only into a new output folder."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import write_json, read_json
from prepare_splits import make_split
from test_revision import fixtures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('revision/my_check'))
    p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    root=a.out.resolve()
    if root.exists():
        p.error('Output exists; choose a new --out directory')
    scripts=Path(__file__).resolve().parent
    project_root=scripts.parents[1]
    root.mkdir(parents=True)
    data=root/'data'
    fixtures(data)
    split=root/'toy.json'
    write_json(split,make_split(data/'REAL','toy',0.5,2026))
    def run(name,*args):
        subprocess.run([sys.executable,str(scripts/name),*map(str,args)],check=True)
    run('train.py','--config',project_root/'configs'/'smoke.json','--split',split,
        '--real_root',data/'REAL','--synth_root',data/'SYNTH','--out_root',root/'runs',
        '--seeds',42,123,'--device','cpu','--workers',a.workers)
    for seed in (42,123):
        run('evaluate.py','--run',root/'runs'/'toy'/f'seed_{seed}',
            '--real_root',data/'REAL','--device','cpu','--workers',a.workers)
    run('compare_seeds.py','--a',root/'runs'/'toy','--b',root/'runs'/'toy',
        '--round_a',0,'--round_b',2,'--bootstrap',200,'--out',root/'paired.json')
    assert read_json(root/'paired.json')['n_seeds']==2
    print('PASS: two seeds, three rounds, held-out evaluation and paired analysis. Fixture scores are not research results.')


if __name__=='__main__':
    main()
