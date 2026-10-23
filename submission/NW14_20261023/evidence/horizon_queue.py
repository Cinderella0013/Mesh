"""Portable endpoint queue readback only; no reconstruction of intermediate queue events."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

HEADER = 'timeNs,node,role,queue,event,packetUid,packetBytes,trackedPackets,actualPackets,trackedBytes,actualBytes,epoch,phase,queuedUids,inflightUids'.split(',')
STATES = ('delivered', 'undelivered_mac_terminal', 'undelivered_censored')
PARTITION = ('delivered', 'undelivered_mac_terminal', 'censored_inflight', 'censored_queued_notinflight', 'censored_notobserved_in_tracked_mac_queue')


def entry_key(row):
    return int(row['node']), row['role'], row['queue']


def uid_set(text):
    values = [int(v) for v in text.split('|')] if text else []
    if any(v < 0 for v in values) or len(values) != len(set(values)):
        raise ValueError('negative or duplicate UID in checkpoint list')
    return set(values)


def validate_checkpoint(row, phase, at):
    if row['event'] != 'checkpoint' or row['phase'] != phase or int(row['timeNs']) != at or int(row['epoch']) != 0:
        raise ValueError('checkpoint event/phase/time/epoch mismatch')
    if row['role'] not in ('AP', 'STA') or int(row['node']) < 0 or not row['queue'] or row['packetUid'] or row['packetBytes']:
        raise ValueError('checkpoint identity or unexpected per-packet fields')
    queued, inflight = uid_set(row['queuedUids']), uid_set(row['inflightUids'])
    counts = [int(row[k]) for k in ('trackedPackets', 'actualPackets', 'trackedBytes', 'actualBytes')]
    if min(counts) < 0 or counts[0] != counts[1] or counts[2] != counts[3] or counts[0] != len(queued):
        raise ValueError('checkpoint tracked/actual/UID count mismatch')
    if not inflight <= queued:
        raise ValueError('inflight is not a subset of queued')
    if phase == 'initial' and (queued or inflight or any(counts)):
        raise ValueError('initial queue checkpoint is not empty')
    return queued, inflight


def extract_horizon(fileobj):
    """Works with seekable ordinary binary files or ZipExtFile; reads header/head and growing tail."""
    fileobj.seek(0)
    raw_header = fileobj.readline()
    if next(csv.reader([raw_header.decode('utf-8-sig').rstrip('\r\n')])) != HEADER:
        raise ValueError('unexpected queue CSV schema')
    initial = {}; head_bytes = len(raw_header)
    while True:
        line = fileobj.readline(); head_bytes += len(line)
        if not line:
            break
        if head_bytes > 1024 * 1024:
            raise ValueError('initial checkpoint prefix exceeds 1MiB')
        values = next(csv.reader([line.decode('utf-8').rstrip('\r\n')]))
        if len(values) != len(HEADER):
            raise ValueError('malformed initial-prefix row')
        row = dict(zip(HEADER, values))
        if row['event'] != 'checkpoint' or row['phase'] != 'initial':
            break
        validate_checkpoint(row, 'initial', 0)
        key = entry_key(row)
        if key in initial:
            raise ValueError('duplicate initial queue entry')
        initial[key] = row
    if not initial or {r for _, r, _ in initial} != {'AP', 'STA'}:
        raise ValueError('missing initial AP/STA queue entries')
    fileobj.seek(0, 2); size = fileobj.tell(); window = min(65536, size); tail_bytes_read = 0
    while True:
        start = size-window; fileobj.seek(start); blob = fileobj.read(window); tail_bytes_read += len(blob)
        if not blob.endswith(b'\n'):
            raise ValueError('queue CSV lacks a complete final newline')
        lines = blob.splitlines()
        if lines:
            lines = lines[1:]  # discard partial first row, or the header when start==0
        records = []
        for values in csv.reader(line.decode('utf-8') for line in lines if line):
            if len(values) != len(HEADER):
                raise ValueError('malformed queue tail row')
            records.append(dict(zip(HEADER, values)))
        horizon = [r for r in records if r['event'] == 'checkpoint' and r['phase'] == 'horizon']
        found = {entry_key(r) for r in horizon}
        first = next((i for i, r in enumerate(records) if r['event'] == 'checkpoint' and r['phase'] == 'horizon'), None)
        boundary_seen = first is not None and (first > 0 or start == 0)
        if set(initial) <= found and boundary_seen:
            break
        if start == 0:
            raise ValueError('missing horizon queue entries or missing final horizon boundary')
        window = min(size, window*2)
    if len(horizon) != len(initial) or found != set(initial):
        raise ValueError('horizon entries differ from initial entries or contain duplicates')
    if records[first:] != horizon:
        raise ValueError('horizon checkpoints are not the final contiguous suffix')
    for row in horizon:
        validate_checkpoint(row, 'horizon', 10_000_000_000)
    return dict(status='PASS', initial_entries=[dict(node=n, role=r, queue=q) for n, r, q in sorted(initial)],
                horizon_entries=horizon, initial_entry_count=len(initial), horizon_entry_count=len(horizon),
                file_bytes=size, head_bytes_read=head_bytes, tail_unique_bytes=window, tail_bytes_read_including_growth=tail_bytes_read,
                tail_start_byte=start, time_ns=10_000_000_000,
                proof='Same initial/horizon entry keys; initial empty at0; final contiguous horizon suffix at10s; tracked==actual counts/bytes; queued UID count exact; inflight subset queued.')


def classify_horizon(snapshot, app):
    """Use own STA queued/inflight sets for the exclusive application UID partition."""
    entries = snapshot['horizon_entries']; nodes = sorted({int(r['node']) for r in entries} | {n for n, _ in app})
    app_nodes={n for n,_ in app}
    if app_nodes!={int(r['node']) for r in entries} or any({r['role'] for r in entries if int(r['node'])==n}!={'AP','STA'} for n in app_nodes):
        raise ValueError('initial/horizon coverage must include AP and STA queues for every application node')
    app_nodes = {n for n, _ in app}
    for role in ('AP', 'STA'):
        if not app_nodes <= {int(r['node']) for r in entries if r['role'] == role}:
            raise ValueError('application node lacks tracked ' + role + ' horizon entries')
    queued = collections.defaultdict(set); inflight = collections.defaultdict(set); observations = collections.defaultdict(list)
    queue_memberships = collections.Counter()
    for row in entries:
        n, role, name = entry_key(row); qs, fs = validate_checkpoint(row, 'horizon', 10_000_000_000)
        queued[n, role].update(qs); inflight[n, role].update(fs)
        for uid in qs:
            observations[uid].append(dict(node=n, role=role, queue=name, inflight=uid in fs))
            if role == 'STA': queue_memberships[n, uid] += 1
    uid_nodes = collections.defaultdict(set)
    for (n, uid), record in app.items():
        if record['status'] not in STATES or (record['received'] is not None) != (record['status'] == 'delivered'):
            raise ValueError('application final state/receive inconsistency')
        uid_nodes[uid].add(n)
        if len(uid_nodes[uid])>1:
            raise ValueError('application UID reused across nodes')
    flows = []; examples = collections.defaultdict(list)
    for n in nodes:
        own_q, own_f = queued[n, 'STA'], inflight[n, 'STA']; c = collections.Counter({k: 0 for k in PARTITION})
        own = {uid: r for (node, uid), r in app.items() if node == n}
        for uid, record in own.items():
            state = record['status']; in_q, in_f = uid in own_q, uid in own_f
            label = state if state != 'undelivered_censored' else 'censored_inflight' if in_f else 'censored_queued_notinflight' if in_q else 'censored_notobserved_in_tracked_mac_queue'
            c[label] += 1; c['application_tx_uids'] += 1
            if state != 'undelivered_censored' and in_q:
                overlap = state + '_also_own_sta_queued'; c[overlap] += 1
                c[state + '_also_own_sta_inflight'] += in_f
                if len(examples[overlap]) < 20:
                    examples[overlap].append(dict(node=n, uid=uid, application=record, queue_observations=observations[uid]))
            seen = observations.get(uid, [])
            c['application_uids_observed_in_own_ap_queue'] += any(v['node'] == n and v['role'] == 'AP' for v in seen)
            c['application_uids_observed_in_other_node_sta_queue'] += any(v['node'] != n and v['role'] == 'STA' for v in seen)
            c['application_uids_observed_in_other_node_ap_queue'] += any(v['node'] != n and v['role'] == 'AP' for v in seen)
            if state == 'undelivered_censored' and not in_q and seen:
                c['censored_not_in_own_sta_but_observed_elsewhere'] += 1
                if len(examples['censored_observed_elsewhere']) < 20:
                    examples['censored_observed_elsewhere'].append(dict(node=n, uid=uid, queue_observations=seen))
        c['own_sta_queued_uids'] = len(own_q); c['own_sta_inflight_uids'] = len(own_f)
        c['own_sta_queued_uids_absent_from_own_application'] = len(own_q-set(own))
        c['own_sta_queued_other_node_application_uids'] = sum(uid in uid_nodes for uid in own_q-set(own))
        c['own_sta_queued_nonapplication_uids'] = sum(uid not in uid_nodes for uid in own_q-set(own))
        c['own_sta_uids_in_multiple_tracked_queues'] = sum(queue_memberships[n, uid] > 1 for uid in own_q)
        if sum(c[k] for k in PARTITION) != len(own):
            raise ValueError('application queue partition is not exclusive/exhaustive')
        for uid in sorted(own_q-set(own))[:10]:
            examples['own_sta_uid_absent_from_own_application'].append(dict(node=n, uid=uid, application_nodes=sorted(uid_nodes.get(uid, [])), queue_observations=observations[uid]))
        flows.append(dict(node=n, **c))
    totals = collections.Counter()
    for f in flows:
        totals.update({k: v for k, v in f.items() if k != 'node'})
    return dict(status='PASS', partition_names=list(PARTITION), counts=dict(totals), flows=flows, examples=dict(examples),
                unique_horizon_queue_uids=len(observations),
                interpretation='Inflight is a subset of queued. Only own STA entries define the censored partition. Not observed in tracked MAC queue is not classified as lost; AP/other-node and delivered/terminal overlaps are separate.')


def horizon_queue_view(native, app):
    """Root integration API. Errors remain data and never silently become not-queued packets."""
    native = Path(native); path = native/'receiver.csv.queue-continuity.csv'
    result = dict(queue_path=str(path), status='ERROR')
    try:
        h = hashlib.sha256(); count = 0
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4*1024*1024), b''):
                h.update(block); count += len(block)
        manifest = json.loads((native/'artifact_hashes.json').read_text(encoding='utf-8-sig'))
        actual, expected = h.hexdigest(), manifest.get(path.name)
        result.update(queue_actual_sha256=actual, queue_expected_sha256=expected, queue_hash_match=actual == expected, queue_bytes_hashed=count)
        with path.open('rb') as f:
            snapshot = extract_horizon(f)
        classification = classify_horizon(snapshot, app)
        result.update(status='PASS' if actual == expected else 'CHECK_FAILED', queue_actual_sha256=actual,
                      queue_expected_sha256=expected, queue_hash_match=actual == expected, queue_bytes_hashed=count,
                      snapshot=snapshot, classification=classification)
        if actual != expected:
            result['error'] = 'queue hash mismatch or absent archived expected hash'
    except Exception as error:
        result['error'] = repr(error)
    return result


def selfcheck():
    def row(n, role, phase, q='', flight=''):
        size = len(q.split('|')) if q else 0
        return dict(zip(HEADER, map(str, [0 if phase == 'initial' else 10_000_000_000, n, role, 'nonqos', 'checkpoint', '', '', size, size, size*1264, size*1264, 0, phase, q, flight])))
    records = [row(0, 'AP', 'initial'), row(0, 'STA', 'initial'), row(1, 'AP', 'initial'), row(1, 'STA', 'initial'),
               row(0, 'AP', 'horizon', '5'), row(0, 'STA', 'horizon', '1|2|3|4|8', '1|3'), row(1, 'AP', 'horizon'), row(1, 'STA', 'horizon')]
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    snapshot = extract_horizon(io.BytesIO(stream.getvalue().encode()))
    import zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('queue.csv', stream.getvalue())
    archive.seek(0)
    with zipfile.ZipFile(archive) as z, z.open('queue.csv') as f:
        assert extract_horizon(f)['horizon_entry_count'] == 4
    wide = records.copy(); wide[5] = row(0, 'STA', 'horizon', '|'.join(map(str, range(20000))), '1|3')
    large = io.StringIO(); writer = csv.DictWriter(large, fieldnames=HEADER, lineterminator='\n'); writer.writeheader(); writer.writerows(wide)
    assert extract_horizon(io.BytesIO(large.getvalue().encode()))['tail_bytes_read_including_growth'] > 65536
    app = {(0, 1): dict(status='delivered', received=2, sent=1), (0, 2): dict(status='undelivered_mac_terminal', received=None, sent=1),
           (0, 3): dict(status='undelivered_censored', received=None, sent=1), (0, 4): dict(status='undelivered_censored', received=None, sent=1),
           (0, 5): dict(status='undelivered_censored', received=None, sent=1), (1, 8): dict(status='undelivered_censored', received=None, sent=1)}
    result = classify_horizon(snapshot, app); c = result['counts']
    assert [c[k] for k in PARTITION] == [1, 1, 1, 1, 2]
    assert c['delivered_also_own_sta_queued'] == c['undelivered_mac_terminal_also_own_sta_queued'] == 1
    assert c['own_sta_queued_other_node_application_uids'] == 1 and c['censored_not_in_own_sta_but_observed_elsewhere'] == 2
    invalid = row(0, 'STA', 'horizon', '1', '2')
    try:
        validate_checkpoint(invalid, 'horizon', 10_000_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError('inflight outside queued must fail')
    invalid = row(0, 'STA', 'horizon', '1'); invalid['actualPackets'] = '0'
    try:
        validate_checkpoint(invalid, 'horizon', 10_000_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError('tracked/actual mismatch must fail')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-folder', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args(); selfcheck()
    if args.selfcheck:
        print('PASS: endpoint extraction, exclusive partition, overlap, other-node/AP cases, subset and count negatives'); return
    if args.native_folder is None:
        parser.error('--native-folder is required unless --selfcheck')
    app = {}
    with (args.native_folder/'receiver.csv.packet-cohort.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = int(row['node']), int(row['packetUid'])
            if key in app:
                raise ValueError('duplicate application UID key')
            app[key] = dict(sent=int(row['sentAtNs']), received=int(row['receivedAtNs']) if row['receivedAtNs'] else None,
                            status=row['status'], duplicates=int(row['duplicateReceives']))
    result = horizon_queue_view(args.native_folder, app)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
