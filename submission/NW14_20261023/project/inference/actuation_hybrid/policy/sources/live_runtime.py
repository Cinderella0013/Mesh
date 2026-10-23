"""Live CPU policy supervisor over the qualified native report/command/ACK bridge."""
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
import common_observation
import native_report
import fresh_calibration
from policy_timing import timed
from native_fixture_runner import linux_path,LOADER,SYSTEM_LIBS,freeze_inputs

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def application_bytes(output,arguments):
    path=output/'receiver.csv.corrected.csv'
    partial=[a.split('=',1)[1] for a in arguments if a.startswith('--developmentStopAtNs=')]
    if partial:
        if len(partial)!=1 or ('THREEWAY_DIAGNOSTIC_STOP_NO_SERVICE_SCORE '+partial[0]) not in (output/'stdout.log').read_text().splitlines():
            raise ValueError('partial native terminal stop evidence missing')
        return None
    with path.open(encoding='utf-8') as f:return sum(int(row['appRxBytes']) for row in csv.DictReader(f))


def host_path(value):
    if sys.platform=='win32' and value.startswith('/mnt/'):
        value=value[5]+':'+value[6:]
    return Path(value)


class ProcessLines:
    """One reader avoids blocking forever in readline while a native job is silent."""
    def __init__(self,command,log_path,*,cwd=None,env=None):
        self.log=Path(log_path).open('x',encoding='utf-8');self.queue=queue.Queue()
        self.process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.log.write(line);self.log.flush();self.queue.put(line)
        finally:
            self.queue.put(None)

    def next(self,timeout):
        try:return self.queue.get(timeout=timeout)
        except queue.Empty as exc:raise TimeoutError('native stdout observation timed out') from exc

    def send(self,value):
        self.process.stdin.write(value+'\n');self.process.stdin.flush()

    def close(self):
        if not self.process.stdin.closed:self.process.stdin.close()
        # EOF aborts the native policy prompt; waiting preserves its real terminal code.
        try:code=self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill();code=self.process.wait(timeout=30)
        self.thread.join(timeout=10);self.process.stdout.close();self.log.close();return code


def next_until(child,deadline,*,poll_seconds=60):
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('live native episode deadline')
        try:return child.next(min(poll_seconds,remaining))
        except TimeoutError:
            # An observation wait expiring is not a terminal job or a restart trigger.
            continue


def verified_binary(reference):
    reference=Path(reference).resolve();receipt=json.loads((reference/'receipt.json').read_text())
    built_only=receipt['status']=='BUILT'
    if built_only:
        receipt=dict(receipt,build=str(reference),source_inputs=receipt['inputs'])
    elif receipt['status']!='TERMINAL' or receipt['exit_code']!=0:
        raise ValueError('successful native job or complete build reference required')
    if 'build' not in receipt:
        original=host_path(receipt['reference']).resolve()
        if original==reference or sha(original/'receipt.json')!=receipt['reference_receipt_sha256']:
            raise ValueError('live job native reference identity changed')
        origin,binary,library_path,validation=verified_binary(original)
        if (host_path(receipt['native_build']).resolve()!=binary.parent.resolve()
            or receipt['binary_sha256']!=origin['binary_sha256']):
            raise ValueError('live job compiled binary differs from verified native reference')
        return receipt,binary,library_path,dict(validation,live_native_reference=str(original))
    build=host_path(receipt['build']);binary=build/'fixture'
    if sha(binary)!=receipt['binary_sha256']:raise ValueError('native binary identity changed')
    build_receipt=json.loads((build/'receipt.json').read_text())
    if build_receipt['status']!='BUILT' or build_receipt['inputs']!=receipt['source_inputs'] or build_receipt['binary_sha256']!=receipt['binary_sha256']:
        raise ValueError('qualified binary/build provenance disagrees')
    expected={};frozen_sources=0;pipeline_sha=None
    for original,digest in receipt['source_inputs'].items():
        path=host_path(original).resolve()
        if path.parent==HERE/'native':
            copied=build/path.name
            if not copied.is_file() or sha(copied)!=digest:raise ValueError('compiled native source snapshot changed: '+path.name)
            expected[linux_path(copied)]=digest;frozen_sources+=1
        elif path==HERE/'native_fixture_runner.py':
            # This historical build driver is provenance, not code executed by
            # the already compiled ELF. The current supervisor is copied anew.
            pipeline_sha=digest
        else:expected[original]=digest
    expected.update(json.loads((build/'dependencies.json').read_text()))
    expected[linux_path(binary)]=receipt['binary_sha256']
    script="import hashlib,json,sys; from pathlib import Path; wanted=json.loads(sys.stdin.read()); bad=[p for p,h in wanted.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]; print(json.dumps({'checked':len(wanted),'bad':bad})); sys.exit(bool(bad))"
    command=[LOADER,'--library-path',SYSTEM_LIBS,'/usr/bin/python3','-I','-B','-c',script]
    if sys.platform=='win32':command=['wsl','-d','Ubuntu','--exec',*command]
    check=subprocess.run(command,input=json.dumps(expected),capture_output=True,text=True,timeout=120)
    if check.returncode:raise ValueError('native binary/source/library dependency changed: '+check.stdout+check.stderr)
    libraries=[p for p in receipt['source_inputs'] if p.endswith('.so')]
    if not libraries:raise ValueError('native linked-library identity absent')
    library_path=libraries[0].rsplit('/',1)[0]+':'+SYSTEM_LIBS
    validation=json.loads(check.stdout)
    validation.update(frozen_native_sources=frozen_sources,historical_pipeline_source_sha256=pipeline_sha,
                      reference_qualification='BUILD_ONLY_NOT_SIMULATION_PASS' if built_only else 'SUCCESSFUL_NATIVE_JOB')
    return receipt,binary,library_path,validation


def run(reference,output,arguments,policy,*,tokens,timeout_seconds=900,resource_profile=None):
    output=Path(output).resolve()
    if HERE not in output.parents or output.exists():raise ValueError('fresh live output inside comparison root required')
    if not 0<timeout_seconds<=3600:raise ValueError('bounded live runtime timeout required')
    if any(arg.startswith(('--output=','--flowMonitorOutput=')) for arg in arguments):raise ValueError('supervisor owns native output paths')
    if resource_profile is not None:
        import campaign_resources
        campaign_resources.check_profile(resource_profile)
        selected=[a for a in arguments if a.startswith('--requiredCpu=')]
        if selected and selected!=['--requiredCpu=2']:raise ValueError('conflicting native CPU requirement')
        if not selected:arguments=[*arguments,'--requiredCpu=2']
    tokens=np.asarray(tokens)
    if tokens.shape!=(15,) or tokens.dtype.kind not in 'iuf' or not np.isfinite(tokens).all() or np.any((tokens<0)|(tokens>1)):
        raise ValueError('fixed fifteen-agent token vector required')
    started=time.monotonic();receipt_reference,binary,library_path,validation=verified_binary(reference)
    output.mkdir(parents=True)
    arguments,frozen_inputs=freeze_inputs(arguments,output)
    def save_receipt(value):
        (output/'receipt.json').write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    receipt=dict(status='STARTED',scope='LIVE_POLICY_DEVELOPMENT_INTERFACE_ONLY',reference=str(Path(reference).resolve()),
        reference_receipt_sha256=sha(Path(reference)/'receipt.json'),binary_sha256=receipt_reference['binary_sha256'],
        native_build=str(binary.parent),dependency_validation=validation,arguments=arguments,tokens=tokens.tolist(),
        frozen_inputs=frozen_inputs,
        policy_model_sha256=getattr(policy,'identity',None),decisions=0,complete_acks=0,
        policy_setup_reported_cpu_ns=getattr(policy,'setup_cpu_ns',None),
        policy_setup_elapsed_ns=getattr(policy,'setup_elapsed_ns',None))
    source_paths=[Path(__file__),HERE/'native_report.py',HERE/'common_observation.py',HERE/'policy_adapters.py']
    source_paths += [HERE/'ganet.py',HERE/'dgn_worker.py',HERE/'candidate_sources/marl_gnn.py',HERE/'candidate_sources/raes.py']
    source_paths += [HERE/'native_fixture_runner.py']
    source_paths += [HERE/'policy_capsule.py',HERE/'policy_timing.py',HERE/'fresh_calibration.py']
    if resource_profile is not None:
        receipt['controller_resources']=resource_profile
        source_paths+=[HERE/'campaign_resources.py']
    receipt['supervisor_sources']={str(p):sha(p) for p in source_paths}
    (output/'sources').mkdir()
    for path in source_paths:
        # Preserve the imported source version before any live commands are collected.
        copy=output/'sources'/path.relative_to(HERE);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
    receipt['clock_info']={name:vars(time.get_clock_info(name)) for name in ('perf_counter','process_time')}
    import policy_capsule
    metadata=policy_capsule.freeze(policy,output/'policy')
    receipt['policy_capsule_manifest_sha256']=sha(output/'policy/artifact_hashes.json')
    receipt['policy_capsule_kind']=metadata['kind']
    receipt['planning_delay_estimand']='MEASURED_ELAPSED_FEATURE_POLICY_POST'
    receipt['timing_boundaries']=dict(feature='report decode/calibration/graph/features',
        decision='input conversion + neural or RPC + validation + pressure/planning/action + evidence preparation',
        post='complete action/phase validation and native command-vector encoding',
        excluded='audit trace-file I/O and final measured-latency field insertion/pipe write',
        external_cpu='reported native worker request-body CPU; worker phases are descriptive parts of RPC, never added to elapsed charge')
    save_receipt(receipt)
    command=[LOADER,'--inhibit-cache','--library-path',library_path,linux_path(binary),
        '--output='+linux_path(output/'receiver.csv'),'--flowMonitorOutput='+linux_path(output/'flow.xml'),*arguments]
    if sys.platform=='win32':
        command=campaign_resources.native_command(command) if resource_profile is not None else ['wsl','-d','Ubuntu','--exec',*command]
    receipt['command']=command;save_receipt(receipt)
    env={k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT')}
    child=None;calibration=None;pending={};acks=set();exit_code=None
    receipt['unavailable_acquisitions']=[]
    try:
        native_started=time.monotonic()
        child=ProcessLines(command,output/'stdout.log',cwd=output,env=env)
        with (output/'decisions.jsonl').open('x',encoding='utf-8') as decisions,(output/'acks.jsonl').open('x',encoding='utf-8') as ack_log:
            while True:
                line=next_until(child,started+timeout_seconds)
                if line is None:break
                if line.startswith('THREEWAY_RESOURCE ') and resource_profile is not None:
                    observed=json.loads(line.split(' ',1)[1])
                    if 'native_resources' in receipt or observed['cpus']!=[2]:raise ValueError('invalid/duplicate native CPU identity')
                    receipt['native_resources']=observed;save_receipt(receipt)
                elif line.startswith('THREEWAY_OBS '):
                    feature_start=time.process_time_ns();feature_elapsed_start=time.perf_counter_ns()
                    report=json.loads(line.split(' ',1)[1])
                    if 'acquisition' in report:
                        calibration=fresh_calibration.load_report(output,report)
                    elif calibration is None:
                        power_path=output/'receiver.csv.sounding.csv'
                        with power_path.open(encoding='utf-8') as f:power_rows=list(csv.DictReader(f))
                        calibration=common_observation.calibrate(power_rows,agents=15,channels=4,epoch=report['availableAtNs']/1e9,
                                                                 source_sha256=sha(power_path))
                        receipt['calibration_digest']=calibration['digest']
                    decoded=native_report.decode(report,calibration,tokens=tokens,decision_at_ns=report['availableAtNs'])
                    feature_cpu=time.process_time_ns()-feature_start
                    feature_elapsed=time.perf_counter_ns()-feature_elapsed_start
                    (channels,evidence),decision_elapsed,decision_cpu=timed(lambda:policy.decide(decoded['observation']))
                    post_start=time.perf_counter_ns();post_cpu_start=time.process_time_ns()
                    channels=np.asarray(channels)
                    if channels.shape!=(15,) or channels.dtype.kind not in 'iu' or np.any((channels<0)|(channels>=4)):
                        raise ValueError('policy returned incomplete/illegal channel vector')
                    external_cpu=evidence.get('external_cpu_ns',0)
                    if type(external_cpu) is not int or external_cpu<0:raise ValueError('invalid external policy CPU accounting')
                    epoch=decoded['epoch']
                    if epoch in pending:raise ValueError('duplicate live observation epoch')
                    phases=evidence.get('phase_costs')
                    if not isinstance(phases,dict) or not phases:raise ValueError('measured policy phases required')
                    for phase in phases.values():
                        if set(phase)!={'elapsed_ns','cpu_ns'} or any(type(v) is not int or v<0 for v in phase.values()):
                            raise ValueError('invalid measured phase cost')
                    phase_elapsed=sum(p['elapsed_ns'] for p in phases.values());phase_cpu=sum(p['cpu_ns'] for p in phases.values())
                    if phase_elapsed>decision_elapsed or phase_cpu>decision_cpu:raise ValueError('phase costs exceed measured decision')
                    encoded_channels=' '.join(map(str,channels.tolist()))
                    post_elapsed=time.perf_counter_ns()-post_start;post_cpu=time.process_time_ns()-post_cpu_start
                    charged_latency=feature_elapsed+decision_elapsed+post_elapsed
                    if 'acquisition' in report:
                        fresh_calibration.validate_binding(report,calibration,report['availableAtNs']+charged_latency)
                    record=dict(epoch=epoch,observed_at_ns=report['observedAtNs'],available_at_ns=report['availableAtNs'],
                        snapshot_digest=decoded['observation']['digest'],report_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        feature_cpu_ns=feature_cpu,decision_cpu_ns=decision_cpu,external_cpu_ns=external_cpu,
                        total_reported_cpu_ns=feature_cpu+decision_cpu+post_cpu+external_cpu,
                        feature_elapsed_ns=feature_elapsed,decision_elapsed_ns=decision_elapsed,
                        post_elapsed_ns=post_elapsed,post_cpu_ns=post_cpu,
                        decision_other_elapsed_ns=decision_elapsed-phase_elapsed,decision_other_cpu_ns=decision_cpu-phase_cpu,
                        charged_latency_ns=charged_latency,channels=channels.tolist(),policy_evidence=evidence)
                    if 'acquisition' in report:
                        record.update(acquisition=report['acquisition'],calibration_digest=calibration['digest'],
                            calibration_source_sha256=calibration['source_sha256'],
                            acquisition_age_at_submit_ns=report['availableAtNs']+charged_latency-calibration['oldest_acquisition_ns'])
                    decisions.write(json.dumps(record,allow_nan=False)+'\n');decisions.flush();pending[epoch]=record
                    child.send(str(epoch)+' '+str(charged_latency)+' '+encoded_channels)
                    receipt['decisions']+=1
                elif line.startswith('THREEWAY_UNAVAILABLE '):
                    unavailable=json.loads(line.split(' ',1)[1])
                    if (set(unavailable)!={'epoch','availableAtNs','reason'} or
                        type(unavailable['epoch']) is not int or not 1<=unavailable['epoch']<=13 or
                        type(unavailable['availableAtNs']) is not int or
                        unavailable['reason']!='radio_not_quiescent' or unavailable['epoch'] in pending or
                        unavailable['epoch'] in [u['epoch'] for u in receipt['unavailable_acquisitions']]):
                        raise ValueError('invalid unavailable acquisition event')
                    receipt['unavailable_acquisitions'].append(unavailable)
                elif line.startswith('THREEWAY_ACK '):
                    ack=json.loads(line.split(' ',1)[1]);epoch=ack['epoch']
                    if epoch not in pending or epoch in acks:raise ValueError('unmatched or duplicate live ACK')
                    if len(ack['channels'])!=15 or len(ack['statuses'])!=15 or ack['availableAtNs']<pending[epoch]['available_at_ns']+pending[epoch]['charged_latency_ns']:
                        raise ValueError('invalid complete live ACK/readback')
                    if any(type(c) is not int or not 0<=c<4 for c in ack['channels']):raise ValueError('invalid acknowledged channels')
                    if any(type(c) is not int or not 0<=c<=4 for c in ack['statuses']):raise ValueError('invalid acknowledged outcomes')
                    if any(status==0 and actual!=target for status,actual,target in zip(ack['statuses'],ack['channels'],pending[epoch]['channels'])):
                        raise ValueError('successful ACK differs from submitted action')
                    ack_log.write(json.dumps(ack,allow_nan=False)+'\n');ack_log.flush();acks.add(epoch);receipt['complete_acks']+=1
        exit_code=child.close();child=None
        if exit_code:
            raise RuntimeError('native process exited with code '+str(exit_code))
        if resource_profile is not None and 'native_resources' not in receipt:
            raise ValueError('native live packet process did not prove required affinity')
        receipt.update(status='TERMINAL',exit_code=exit_code,elapsed_seconds=time.monotonic()-started,
                       native_elapsed_seconds=time.monotonic()-native_started,
                       epochs_without_complete_ack=sorted(set(pending)-acks))
        if exit_code==0:
            receipt['application_rx_bytes']=application_bytes(output,arguments)
        save_receipt(receipt)
    except Exception as exc:
        code=child.close() if child is not None else exit_code
        receipt.update(status='FAILED_PIPELINE',terminal=True,exit_code=code,error=repr(exc),elapsed_seconds=time.monotonic()-started)
        save_receipt(receipt);raise
    finally:
        hashes={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p!=output/'artifact_hashes.json'}
        (output/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    return receipt
