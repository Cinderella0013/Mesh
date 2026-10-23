"""Source-derived repaired Routing graphs and bounded native KL/optimizer evidence."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent / 'dgn_cpu_preparation_20261001'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load('target_helpers', PREP / 'routing_target_probe.py')
td = load('td_helpers', PREP / 'routing_td_probe.py')


def extract_regularizer(source):
    body = ast.parse(source).body
    starts = [i for i, n in enumerate(body) if helpers.full.assignment(n, 'Inr')]
    ends = [i for i, n in enumerate(body) if helpers.full.assignment(n, 'model_r')]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError('Regularizer boundaries missing or ambiguous')
    nodes = body[starts[0]:ends[0]+1]
    for node in nodes:
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in (
                    'compile', 'fit', 'train_on_batch', 'set_weights', 'get_updates', 'apply_gradients'):
                raise ValueError('Mutation inside regularizer assembly')
    return ''.join(source.splitlines(keepends=True)[nodes[0].lineno-1:nodes[-1].end_lineno])


def kl_reference(labels, predictions):
    """Keras multi-output loss: sum outputs, mean batch/head/query, sum keys."""
    total = 0.
    if len(labels) != len(predictions):
        raise ValueError('Output count differs')
    for label, pred in zip(labels, predictions):
        terms = []
        for lb, pb in zip(label, pred):
            for lh, ph in zip(lb, pb):
                for lr, pr in zip(lh, ph):
                    terms.append(sum(max(1e-7, min(1., p))*math.log(
                        max(1e-7, min(1., p))/max(1e-7, min(1., q))) for p, q in zip(lr, pr)))
        if not terms:
            raise ValueError('Empty attention batch')
        total += sum(terms)/len(terms)
    return total


def execute(directory, name, namespace):
    path = Path(directory) / name
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)


def build(directory, variant, seed=0, agents=20):
    """Native child only; no environment or optimizer mutation on construction."""
    if sys.version_info[:2] != (3, 7) or struct.calcsize('P') != 8 or not __debug__:
        raise RuntimeError('Python3.7 x64 required')
    if sys.prefix == sys.base_prefix or not sys.flags.isolated:
        raise RuntimeError('Isolated venv required')
    if 'include-system-site-packages = false' not in (Path(sys.prefix)/'pyvenv.cfg').read_text().lower():
        raise RuntimeError('System packages must be excluded')
    versions = {p['name'].lower(): p['version'] for p in helpers.probe.installed_packages(sys.path)}
    for name, version in dict(helpers.probe.PINS, h5py='2.10.0', protobuf='3.20.3').items():
        if versions.get(name) != version:
            raise RuntimeError('Package pin mismatch: '+name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    import numpy as np
    import tensorflow as tf
    if tf.config.get_visible_devices('GPU'):
        raise RuntimeError('GPU visible')
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if agents not in (15,20):
        raise ValueError('Only verified Routing20 or Wi-Fi15 interfaces supported')
    ns = dict(np=np, tf=tf, n_data=agents, neighbors=4, len_feature=35, action_space=4)
    with tf.device('/CPU:0'):
        for name in ('model_definitions.py', 'online_graph.py', 'target_graph.py', 'tau.py', 'gamma.py'):
            execute(directory, name, ns)
        if variant == 'regularized':
            execute(directory, 'regularizer_graph.py', ns)
    return ns


def prepare(directory, variant):
    source = helpers.probe.routing_source(dict(helpers.probe.VARIANTS)[variant]).read_text(encoding='utf-8-sig')
    definitions, graph = helpers.full.extract_online(source)
    target, sync, tau = helpers.extract_target_and_sync(source)
    texts = dict(original_model_definitions=definitions,
                 model_definitions=helpers.repair.repair_extract(definitions),
                 online_graph=graph, target_graph=target, soft_update=sync, tau=tau)
    if variant == 'regularized':
        texts['regularizer_graph'] = extract_regularizer(source)
    texts.update({k[:-3]: v for k, v in td.extract_td(source).items() if k in ('gamma.py', 'td_target.py')})
    for name, value in texts.items():
        (Path(directory)/(name+'.py')).write_text(value, encoding='utf-8')


def model_inputs(np, observations, masks):
    selector = np.zeros((len(observations), 1, 4), dtype='float32')
    selector[:, 0, 0] = 1
    result = []
    for j in range(observations.shape[1]):
        result.extend([observations[:, j], masks[:, j]])
    return result+[selector]


def native_run(variant, directory):
    directory = Path(directory)
    control = helpers.probe.controls()
    summary = dict(status='STARTED', stage='BUILD', variant=variant, optimizer_steps=0,
                   target_soft_updates=0, environment='NOT_RUN', training='SINGLE_BATCH_DIAGNOSTIC',
                   regularized_aggregation='UPSTREAM_RAW_SCORES_PRESERVED', fixture_seed=0)
    def save():
        control.write_json(directory/'summary.json', summary)
    def weights(label, model):
        values = model.get_weights()
        np.savez(directory/(label+'.npz'), *values)
        if not all(np.isfinite(w).all() for w in values):
            raise ValueError('Nonfinite weights: '+label)
        return values
    def equal(a, b):
        return all(np.array_equal(x, y) for x, y in zip(a, b))
    def gradients(graph, labels, loss):
        with graph.outputs[0].graph.as_default():
            objective = tf.add_n([K.mean(loss(K.constant(y), output)) for y, output in zip(labels, graph.outputs)])
            grads = K.gradients(objective, graph.trainable_weights)
            if any(g is None for g in grads):
                raise ValueError('Disconnected gradient')
            return K.function(graph.inputs, [objective]+grads)
    def grad_check(function, inputs, label):
        values = function(inputs)
        if not all(np.isfinite(v).all() for v in values) or not any(np.any(v != 0) for v in values[1:]):
            raise ValueError('Nonfinite or all-zero gradients: '+label)
        np.savez(directory/(label+'_gradients.npz'), *values)
        return float(values[0])
    try:
        save()
        ns = build(directory, variant)
        np, tf, K, Adam = ns['np'], ns['tf'], ns['K'], ns['Adam']
        from keras.losses import kullback_leibler_divergence, mean_squared_error
        model, target = ns['model'], ns['model_t']
        online_ids, target_ids = [id(w) for w in model.weights], [id(w) for w in target.weights]
        assert len(set(online_ids)) == len(set(target_ids)) == 22
        assert not set(online_ids) & set(target_ids)
        pairs = [(online_ids.index(id(a)), target_ids.index(id(b))) for component in helpers.COMPONENTS
                 for a, b in zip(ns[component].weights, ns[component+'_t'].weights)]
        assert sorted(a for a,b in pairs) == sorted(b for a,b in pairs) == list(range(22))
        before, target_before = weights('online_before', model), weights('target_before', target)
        obs = np.linspace(-10, 10, 1400, dtype='float32').reshape(2,20,35)
        masks = np.zeros((2,20,4,20), dtype='float32')
        for j in range(20):
            for slot, neighbor in enumerate(helpers.full.neighbor_indices(j)):
                masks[:,j,slot,neighbor] = 1
        inputs, next_inputs = model_inputs(np, obs, masks), model_inputs(np, 3*obs+2, masks)
        np.savez(directory/'fixture.npz', observations=obs, next_observations=3*obs+2, masks=masks)
        if variant == 'regularized':
            summary['stage'] = 'REGULARIZER_KL'
            save()
            regularizer = ns['model_r']
            rid = [id(w) for w in regularizer.trainable_weights]
            assert len(rid) == 16 and set(rid) <= set(online_ids) and not set(rid) & set(target_ids)
            attention = regularizer.predict(inputs, batch_size=2, verbose=0)
            frozen = regularizer.predict(next_inputs, batch_size=2, verbose=0)
            for label, values in (('current_attention', attention), ('frozen_online_next_attention', frozen)):
                array = np.stack(values, axis=1)
                assert array.shape == (2,20,8,4,4) and np.isfinite(array).all()
                assert np.allclose(array.sum(axis=-1), 1., atol=1e-6)
                np.savez(directory/(label+'.npz'), *values)
            kl_function = gradients(regularizer, frozen, kullback_leibler_divergence)
            native_kl = grad_check(kl_function, inputs, 'kl_before_td')
            reference_kl = kl_reference([v.tolist() for v in frozen], [v.tolist() for v in attention])
            assert math.isclose(native_kl, reference_kl, rel_tol=1e-3, abs_tol=1e-6), (native_kl,reference_kl)
            assert equal(before, model.get_weights()) and equal(target_before, target.get_weights())
            control.write_json(directory/'task2.json', dict(status='PASS', optimizer_steps=0,
                attention_shape=[2,20,8,4,4], normalized=True, shared_online_variables=16,
                gradient_variables=16, native_kl=native_kl, reference_kl=reference_kl,
                direction='KL(online_next_before_TD || online_current)', reduction='SUM_OUTPUTS_MEAN_BATCH_HEAD_QUERY',
                online_unchanged=True, target_unchanged=True))
        summary['stage'] = 'TD_OPTIMIZER'
        save()
        q = model.predict(inputs, batch_size=2, verbose=0)
        tq = target.predict(next_inputs, batch_size=2, verbose=0)
        actions = np.array([[(j+k)%4 for j in range(20)] for k in range(2)])
        rewards = np.array([[-.2 if j%3 else 10. for j in range(20)] for k in range(2)],dtype='float32')
        dones = np.array([[(j+k)%2 == 0 for j in range(20)] for k in range(2)])
        ns.update(batch=[0,1], q_values=[v.copy() for v in q], target_q_values=tq,
                  actions=actions, rewards=rewards, dones=dones)
        execute(directory, 'td_target.py', ns)
        labels = ns['q_values']
        for j in range(20):
            for k in range(2):
                for a in range(4):
                    expected = rewards[k,j]+(0 if dones[k,j] else .98*np.max(tq[j][k])) if a == actions[k,j] else q[j][k,a]
                    assert np.isclose(labels[j][k,a], expected)
        np.savez(directory/'td_targets.npz', *labels)
        td_function = gradients(model, labels, mean_squared_error)
        grad_check(td_function, inputs, 'td_before_update')
        model.compile(optimizer=Adam(lr=.0001 if variant == 'regularized' else .001), loss='mse')
        losses = model.train_on_batch(inputs, labels)
        summary['optimizer_steps'] += 1
        assert np.isfinite(losses).all()
        after_td = weights('online_after_td', model)
        assert not equal(before, after_td) and equal(target_before, target.get_weights())
        if variant == 'regularized':
            summary['stage'] = 'KL_OPTIMIZER'
            save()
            grad_check(kl_function, inputs, 'kl_after_td')
            regularizer.compile(optimizer=Adam(lr=.000003), loss='kullback_leibler_divergence')
            kl_losses = regularizer.train_on_batch(inputs, frozen)
            summary['optimizer_steps'] += 1
            assert np.isfinite(kl_losses).all()
            after_kl = weights('online_after_kl', model)
            assert not equal(after_td, after_kl)
            assert all(np.array_equal(a,b) for i,(a,b) in enumerate(zip(after_td,after_kl)) if online_ids[i] not in rid)
            assert equal(target_before, target.get_weights())
            summary['kl_losses'] = [float(v) for v in kl_losses]
        summary['stage'] = 'SOFT_SYNC'
        save()
        online_final = weights('online_final', model)
        execute(directory, 'soft_update.py', ns)
        summary['target_soft_updates'] += 1
        target_final = weights('target_final', target)
        errors = [float(np.max(np.abs(target_final[b]-(.01*online_final[a]+.99*target_before[b])))) for a,b in pairs]
        assert max(errors) == 0 and not equal(target_before, target_final)
        assert equal(online_final, model.get_weights())
        summary.update(status='PASS', stage='FINISHED', td_losses=[float(v) for v in losses],
                       soft_sync_max_abs_error=max(errors), target_unchanged_before_sync=True,
                       td_gradient_variables=22, regularizer_weight_ownership_verified=variant == 'regularized')
        control.write_json(directory/'task3.json', summary)
    except BaseException:
        summary.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=['baseline','regularized'])
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3,12) and __debug__ and sys.flags.isolated
    assert 1 <= args.max_seconds <= 1200
    control = helpers.probe.controls()
    output = control.new_output(args.output, ROOT/'model_checks')
    inputs = [Path(__file__), ROOT/'check_model_stage.py', args.runtime.resolve(strict=True)]
    before = {str(p): control.sha(p) for p in inputs}
    receipt = dict(status='STARTED', variant=args.variant, authorization='ACTIVE_GOAL_ALL_EIGHT_TASKS',
                   started_at=datetime.now(timezone.utc).isoformat(), hashes_before=before)
    control.write_json(output/'source_before.json',control.check_identity())
    control.write_json(output/'receipt.json',receipt)
    code = 1
    try:
        prepare(output,args.variant)
        child = "import importlib.util;s=importlib.util.spec_from_file_location('stage',"+repr(str(Path(__file__).resolve()))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.native_run("+repr(args.variant)+','+repr(str(output))+')'
        command = [str(args.runtime),'-I','-B','-u','-c',child]
        status, code = control.run_child(command, output, args.max_seconds)
        receipt.update(process_status=status, exit_code=code, command=command)
        if code == 0:
            assert json.loads((output/'task3.json').read_text())['status'] == 'PASS'
        receipt['status'] = 'PASS' if code == 0 else 'FAILED'
    except BaseException:
        receipt.update(status='SUPERVISOR_FAILED',error=traceback.format_exc())
        code = 1
    finally:
        control.write_json(output/'source_after.json',control.check_identity())
        after = {str(p): control.sha(p) for p in inputs}
        assert before == after
        receipt.update(hashes_after=after,finished_at=datetime.now(timezone.utc).isoformat())
        control.write_json(output/'receipt.json',receipt)
        control.write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):control.sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(receipt['status']+': '+str(output))
    return code


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.exit(main())
