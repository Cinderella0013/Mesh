"""Original TD/replay logic with finite list fixtures and a stdlib np.max adapter."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('routing_controls', ROOT / 'routing_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
REPLAY = probe.routing_source('ReplayBuffer_v2.py')


def unique(nodes):
    if len(nodes) != 1:
        raise ValueError('Missing/ambiguous source boundary')
    return nodes[0]


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def extract_td(source):
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    gamma = unique([n for n in tree.body if assignment(n, 'GAMMA')])
    if ast.literal_eval(gamma.value) != .98:
        raise ValueError('GAMMA changed')
    call = unique([n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and ast.dump(n.value.func) == ast.dump(ast.parse('buff.add', mode='eval').body)])
    if ast.dump(call) != ast.dump(ast.parse('buff.add(obs, act, next_obs, reward, done, adj)').body[0]):
        raise ValueError('Replay positional call changed')
    names = ['states','actions','rewards','new_states','dones','adj']
    start = unique([n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Tuple) and [getattr(t, 'id', None) for t in n.targets[0].elts] == names])
    blocks = [n.body for n in ast.walk(tree) if isinstance(getattr(n, 'body', None), list) and start in n.body]
    body = unique(blocks)
    i = body.index(start)
    assembly = body[i:i+3]
    if len(assembly) != 3 or not isinstance(assembly[1], ast.For) or not isinstance(assembly[2], ast.For):
        raise ValueError('Replay assembly boundaries changed')
    td = unique([n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                 and n.target.id == 'k' and ast.dump(n.iter) == ast.dump(ast.parse('range(len(batch))',mode='eval').body)])
    for node in assembly + [td]:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                allowed = (isinstance(n.func, ast.Name) and n.func.id in ('range','len')) or (
                    isinstance(n.func, ast.Attribute) and (n.func.attr == 'append' or
                    (isinstance(n.func.value, ast.Name) and n.func.value.id == 'np' and n.func.attr == 'max')))
                if not allowed:
                    raise ValueError('Unexpected call in TD/replay extraction')
    def snippet(nodes):
        return textwrap.dedent(''.join(lines[nodes[0].lineno-1:nodes[-1].end_lineno]))
    return {'gamma.py':snippet([gamma]),'replay_add.py':snippet([call]),
            'batch_assembly.py':snippet(assembly),'td_target.py':snippet([td])}


def replay_extract():
    source = REPLAY.read_text(encoding='utf-8-sig')
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    cls = unique([n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ReplayBuffer'])
    imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    return ''.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in imports+[cls])


def make_fixture():
    transitions = []
    for k in range(4):
        obs = [[(k*1000+j*40+f)/100. for f in range(35)] for j in range(20)]
        masks = lambda offsets: [[[float(col == (j+offset)%20) for col in range(20)]
                                 for offset in offsets] for j in range(20)]
        transitions.append({'obs':obs,'next_obs':[[x+.5 for x in row] for row in obs],
                            'act':[(j+k)%4 for j in range(20)],
                            'reward':[(-2.5,0.,1.25,10.)[(j+k)%4] for j in range(20)],
                            'done':[(k//2+j)%2 == 0 for j in range(20)],
                            'adj':masks((0,1,2,3)),'next_adj':masks((0,4,5,6))})
    patterns = ((-20.,-10.,-8.,-5.),(1.,8.,3.,4.),(2.,-3.,5.,0.),(0.,0.,0.,0.))
    return {'transitions':transitions,
            'q_before':[[[float(1000+j*100+k*10+a) for a in range(4)] for k in range(4)] for j in range(20)],
            'target_q':[[[x+j*.05 for x in patterns[k]] for k in range(4)] for j in range(20)]}


def run_variant(variant, directory):
    output = Path(directory)
    control = probe.controls()
    summary = {'variant':variant,'status':'STARTED','stage':'GUARD','agents':20,'transitions':4,'actions':4,
               'forward_batches':0,'backward_evaluations':0,'optimizer_steps':0,'model':'NOT_RUN',
               'environment':'NOT_RUN','training':'NOT_RUN','replay_sampling':'NOT_RUN',
               'adapter':'np.max -> stdlib max; finite Python list fixtures; np.asarray omitted',
               'python':sys.version,'executable':sys.executable}
    before = set(sys.modules)
    def save():
        control.write_json(output/'summary.json',summary)
    def execute(name, namespace):
        path = output/name
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
    try:
        save()
        if sys.version_info[:2] != (3,12) or not __debug__ or not sys.flags.isolated:
            raise RuntimeError('Isolated Python3.12 with assertions required')
        fixture = json.loads((output/'fixture.json').read_text(encoding='utf-8'))
        namespace = {}
        execute('replay_buffer_extract.py',namespace)
        namespace['buff'] = namespace['ReplayBuffer'](4)
        summary['stage'] = 'REPLAY_INSERT_AND_ASSEMBLY'
        save()
        for transition in fixture['transitions']:
            namespace.update({name:copy.deepcopy(transition[name]) for name in ('obs','next_obs','act','reward','done','adj')})
            execute('replay_add.py',namespace)
        batch = list(namespace['buff'].buffer)
        expected_rows = [[t[name] for name in ('obs','act','next_obs','reward','done','adj')] for t in fixture['transitions']]
        if [list(row) for row in batch] != expected_rows or len(batch) != 4:
            raise ValueError('Replay positional payload mismatch')
        control.write_json(output/'replay_rows.json',batch)
        namespace.update(batch=batch,n_data=20,vec=[[1.,0.,0.,0.]])
        execute('gamma.py',namespace)
        execute('batch_assembly.py',namespace)
        assembled = {name:namespace[name] for name in ('states','new_states','actions','rewards','dones')}
        control.write_json(output/'assembled_inputs.json',assembled)
        reused, missing_next = 0,0
        for k,t in enumerate(fixture['transitions']):
            if assembled['actions'][k] != t['act'] or assembled['rewards'][k] != t['reward'] or assembled['dones'][k] != t['done']:
                raise ValueError('Replay action/reward/done slot mismatch')
            for j in range(20):
                if assembled['states'][j*2][k] != t['obs'][j] or assembled['new_states'][j*2][k] != t['next_obs'][j]:
                    raise ValueError('Current/next observation slot mismatch')
                if assembled['states'][j*2+1][k] != t['adj'][j] or assembled['new_states'][j*2+1][k] != t['adj'][j]:
                    raise ValueError('Upstream current-adjacency reuse changed')
                reused += 1
                missing_next += assembled['new_states'][j*2+1][k] != t['next_adj'][j]
            if assembled['states'][40][k] != namespace['vec'] or assembled['new_states'][40][k] != namespace['vec']:
                raise ValueError('Selector assembly mismatch')
        if reused != 80 or missing_next != 80:
            raise ValueError('Changed-neighbor fixture is vacuous')
        calls = []
        def maximum(values):
            if len(values) != 4 or not all(math.isfinite(x) for x in values):
                raise ValueError('Expected finite target Q vector of four actions')
            calls.append(values[:])
            return max(values)
        namespace.update(q_values=copy.deepcopy(fixture['q_before']),target_q_values=copy.deepcopy(fixture['target_q']),
                         np=SimpleNamespace(max=maximum))
        summary['stage'] = 'TD_TARGET'
        save()
        execute('td_target.py',namespace)
        selected, unselected, terminal, bootstrap = 0,0,0,0
        max_error = 0.
        for k,t in enumerate(fixture['transitions']):
            for j in range(20):
                action,done,reward = t['act'][j],t['done'][j],t['reward'][j]
                expected = reward if done else reward+.98*max(fixture['target_q'][j][k])
                actual = namespace['q_values'][j][k][action]
                if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('TD target formula/index mismatch')
                max_error = max(max_error,abs(actual-expected))
                selected += 1
                terminal += done
                bootstrap += not done
                for a in range(4):
                    if a != action:
                        if namespace['q_values'][j][k][a] != fixture['q_before'][j][k][a]:
                            raise ValueError('Unselected action Q changed')
                        unselected += 1
        if len(calls) != bootstrap or namespace['target_q_values'] != fixture['target_q']:
            raise ValueError('Terminal bootstrap call or target Q mutation')
        control.write_json(output/'q_after.json',namespace['q_values'])
        forbidden = {'numpy','tensorflow','keras','torch'} & (set(sys.modules)-before)
        if forbidden:
            raise RuntimeError('Native modules imported: '+str(forbidden))
        summary.update(status='COMPLETED_PENDING_REVIEW',stage='FINISHED',gamma=namespace['GAMMA'],
                       selected_action_checks=selected,unselected_action_checks=unselected,terminal_checks=terminal,
                       bootstrap_checks=bootstrap,maximum_calls=len(calls),td_max_abs_error=max_error,
                       target_q_unchanged=True,observation_slots_match=True,current_adj_reused=reused,
                       next_adj_fixture_not_consumed=missing_next,next_state_adjacency='CURRENT_REPLAY_MASK_REUSED',native_imports=False)
    except Exception:
        summary.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=30)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 30:
        parser.error('--max-seconds must be 1..30 total')
    if sys.version_info[:2] != (3,12) or not __debug__ or not sys.flags.isolated:
        raise RuntimeError('Isolated Python3.12 with assertions required')
    control = probe.controls()
    audit = control.check_identity()
    output = control.new_output(args.output,ROOT/'routing_td_checks')
    inputs = [Path(__file__).resolve(),ROOT/'check_routing_td_probe.py',ROOT/'routing_probe.py',
              ROOT/'compat/torch_surviving_cpu_v1/main.py',probe.REFERENCE/'check_reference.py',probe.REFERENCE/'manifest.json',
              REPLAY,Path(sys.executable)] + [probe.routing_source(filename) for _,filename in probe.VARIANTS]
    hashes = {str(path):control.sha(path) for path in inputs}
    receipt = {'status':'STARTED','scope':'ORIGINAL_TD_REPLAY_LOGIC_STDLIB_ADAPTERS',
               'execution_owner':'ASSISTANT_USER_AUTHORIZED','authorization':'yes approve, 4transitions20agents, forward/backward/optimizer0,30s,3October2026',
               'output':str(output),'runtime':sys.executable,'hashes_before':hashes,'max_seconds_total':args.max_seconds,
               'started_at_utc':datetime.now(timezone.utc).isoformat(),'forward_batches':0,'backward_evaluations':0,'optimizer_steps':0,
               'model':'NOT_RUN','environment':'NOT_RUN','training':'NOT_RUN','ns3':'NOT_RUN','scientific_gates':'CLOSED','variants':[]}
    control.write_json(output/'source_before.json',audit)
    control.write_json(output/'run_receipt.json',receipt)
    deadline,exit_code = time.monotonic()+args.max_seconds,1
    try:
        for variant,filename in probe.VARIANTS:
            directory = control.new_output(output/variant,output)
            extracts = extract_td(probe.routing_source(filename).read_text(encoding='utf-8-sig'))
            extracts['replay_buffer_extract.py'] = replay_extract()
            for name,source in extracts.items():
                (directory/name).write_text(source,encoding='utf-8')
            control.write_json(directory/'fixture.json',make_fixture())
            child = "import importlib.util; s=importlib.util.spec_from_file_location('td',"+repr(str(Path(__file__).resolve()))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.run_variant("+repr(variant)+","+repr(str(directory))+")"
            command = [sys.executable,'-I','-B','-u','-c',child]
            remaining = deadline-time.monotonic()
            status,code = control.run_child(command,directory,remaining) if remaining>0 else ('NOT_STARTED_DEADLINE',124)
            row = {'variant':variant,'process_status':status,'exit_code':code,'command':command,
                   'extract_sha256':{name:control.sha(directory/name) for name in extracts}}
            if (directory/'summary.json').exists():
                row['summary'] = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            receipt['variants'].append(row)
            control.write_json(output/'run_receipt.json',receipt)
            if code == 0:
                s = row.get('summary',{})
                if s.get('status') != 'COMPLETED_PENDING_REVIEW' or s.get('selected_action_checks') != 80 or s.get('unselected_action_checks') != 240 or s.get('terminal_checks') != 40 or s.get('bootstrap_checks') != 40 or s.get('current_adj_reused') != 80 or s.get('next_adj_fixture_not_consumed') != 80 or s.get('native_imports') is not False or any(s.get(k) != 0 for k in ('forward_batches','backward_evaluations','optimizer_steps')):
                    raise ValueError('Missing approved TD/replay evidence')
        codes = [row['exit_code'] for row in receipt['variants']]
        exit_code = 124 if 124 in codes else (0 if codes==[0,0] else 1)
        receipt['status'] = 'COMPLETED_PENDING_REVIEW' if exit_code==0 else ('TIMEOUT' if exit_code==124 else 'FAILED_PENDING_REVIEW')
    except KeyboardInterrupt:
        receipt['status'],exit_code = 'INTERRUPTED',130
    except Exception:
        receipt.update(status='SUPERVISOR_FAILED',supervisor_error=traceback.format_exc())
        exit_code = 1
    finally:
        try:
            control.write_json(output/'source_after.json',control.check_identity())
            receipt['hashes_after'] = {str(path):control.sha(path) for path in inputs}
            if hashes != receipt['hashes_after']:
                raise ValueError('Probe/source/runtime identity changed')
        except Exception:
            receipt.update(status='IDENTITY_FAILED',identity_error=traceback.format_exc())
            exit_code = 1
        receipt.update(exit_code=exit_code,finished_at_utc=datetime.now(timezone.utc).isoformat())
        control.write_json(output/'run_receipt.json',receipt)
    print('{}: {}'.format(receipt['status'],output))
    return exit_code


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.exit(main())
