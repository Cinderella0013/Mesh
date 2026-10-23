"""Compile/run isolated native fixtures using accepted read-only ns-3 libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
LOADER = '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
SYSTEM_LIBS = '/usr/lib/x86_64-linux-gnu'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linux_path(path):
    text = str(path)
    return '/mnt/' + text[0].lower() + text[2:].replace('\\', '/') if len(text)>1 and text[1]==':' else text


def host_path(value):
    if sys.platform=='win32' and value.startswith('/mnt/'):
        value=value[5]+':'+value[6:]
    return Path(value)


def freeze_inputs(arguments,output):
    """All native file arguments execute from fresh per-job copies, not originals."""
    output=Path(output).resolve()
    if HERE not in output.parents or not output.is_dir():raise ValueError('owned output directory required')
    keys={'positionsFile','loadsFile','qValuesFile','demandScheduleFile'}
    selected={}
    for arg in arguments:
        key,separator,value=arg.removeprefix('--').partition('=')
        if arg.startswith('--') and key in keys:
            if not separator or not value or key in selected:raise ValueError('one explicit file argument required: '+key)
            source=host_path(value).resolve()
            if not source.is_file():raise ValueError('native input is not a file: '+str(source))
            selected[key]=source
    identities={};rewritten=list(arguments)
    if selected:
        (output/'inputs').mkdir(exist_ok=False)
        for key,source in selected.items():
            copied=output/'inputs'/(key+'.csv');data=source.read_bytes();copied.write_bytes(data)
            digest=hashlib.sha256(data).hexdigest()
            if sha(source)!=digest:raise ValueError('native input changed during freeze')
            identities[key]=dict(original_path=str(source),copied_path=str(copied),sha256=digest)
        for index,arg in enumerate(arguments):
            key=arg.removeprefix('--').partition('=')[0]
            if key in selected and arg.startswith('--'):
                rewritten[index]='--'+key+'='+linux_path(output/'inputs'/(key+'.csv'))
    return rewritten,identities


def fixture_commands(reference, obj, binary, source, dependency_file):
    compile_command, link_command = [list(command) for command in reference]
    compile_command[compile_command.index('-o')+1] = str(obj)
    compile_command[compile_command.index('-c')+1] = str(source)
    compile_command.extend(['-MMD','-MF',str(dependency_file)])
    objects=[i for i,arg in enumerate(link_command) if arg.endswith('.o')]
    if len(objects)!=1:
        raise ValueError('expected exactly one reference object file')
    link_command[objects[0]] = str(obj)
    link_command[link_command.index('-o')+1] = str(binary)
    return compile_command, link_command


def run(source_name, output, arguments, stdin_path=None,resource_profile=None,timeout_seconds=900):
    output = Path(output).resolve()
    if HERE not in output.parents or output.exists():
        raise ValueError('fresh output inside comparison root required')
    if Path(source_name).name != source_name or not (HERE/'native'/source_name).is_file():
        raise ValueError('known fixture source name required')
    if stdin_path is not None:
        stdin_path=Path(stdin_path).resolve()
        if HERE not in stdin_path.parents or not stdin_path.is_file():
            raise ValueError('policy input must be an existing comparison-root file')
    if resource_profile is not None:
        sys.path.insert(0,str(HERE))
        import campaign_resources
        if sys.platform=='win32':campaign_resources.check_profile(resource_profile)
        elif resource_profile is not True or sorted(os.sched_getaffinity(0))!=[2]:
            raise ValueError('actual native supervisor affinity differs')
        if source_name!='wifi_threeway.cc':raise ValueError('resource profile requires qualified packet source')
        selected=[a for a in arguments if a.startswith('--requiredCpu=')]
        if selected and selected!=['--requiredCpu=2']:raise ValueError('conflicting native CPU requirement')
        if not selected:arguments=[*arguments,'--requiredCpu=2']
    if sys.platform == 'win32':
        command = ['wsl', '-d', 'Ubuntu', '--exec', LOADER, '--library-path', SYSTEM_LIBS,
                   '/usr/bin/python3', '-I', '-B', linux_path(Path(__file__)),
                   '--source', source_name, '--output', linux_path(output)]
        if resource_profile is not None:
            command=campaign_resources.wsl_prefix()+command[7:]
            command+=['--campaign-resources']
        if stdin_path is not None:command+=['--stdin-input',linux_path(stdin_path)]
        command+=['--',*arguments]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError('native fixture pipeline failed: ' + result.stdout + result.stderr)
        return json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    save = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    receipt = dict(status='STARTED', scope='NATIVE_DEVELOPMENT_MEASUREMENT_ONLY', arguments=arguments,
                   stdin_path=str(stdin_path) if stdin_path else None,
                   stdin_sha256=sha(stdin_path) if stdin_path else None)
    if resource_profile is not None:
        receipt['supervisor_resources']=dict(pid=os.getpid(),cpus=sorted(os.sched_getaffinity(0)))
        copied=output/'resource_supervisor.py';copied.write_bytes((HERE/'campaign_resources.py').read_bytes())
        receipt['resource_supervisor_source_sha256']=sha(copied)
    save(output/'receipt.json', receipt)
    arguments,frozen_inputs=freeze_inputs(arguments,output)
    receipt.update(executed_arguments=arguments,frozen_inputs=frozen_inputs)
    if stdin_path is not None:
        copied=output/'policy_stdin.txt';copied.write_bytes(stdin_path.read_bytes())
        if sha(copied)!=receipt['stdin_sha256']:raise ValueError('policy input changed while being copied')
        receipt['executed_stdin_path']=str(copied);stdin_path=copied
    save(output/'receipt.json',receipt)
    external_path = HERE.parent/'library/external-current.json'
    external = json.loads(external_path.read_text())
    commands = external['commands']
    libraries = [Path(p) for p in commands[1] if p.endswith('.so')]
    for path in libraries:
        if sha(path) != external['files'][str(path)]:
            raise ValueError('accepted library changed: ' + str(path))
    source = HERE/'native'/source_name
    headers = list(sorted((HERE/'native').glob('*.h')))
    inputs = {str(p):sha(p) for p in [source, *headers, Path(__file__), external_path, *libraries]}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    build = HERE/'native/builds'/key
    binary, obj = build/'fixture', build/'fixture.o'
    libdir = str(libraries[0].parent)
    prefix = [LOADER, '--inhibit-cache', '--library-path', libdir+':'+SYSTEM_LIBS]
    env = {k:v for k,v in os.environ.items() if k not in ['LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT']}
    # Compiler subprocesses also need the recovered system libc, not /lib's stale one.
    compiler_env = dict(env, LD_LIBRARY_PATH=SYSTEM_LIBS)
    started=time.perf_counter()
    if not binary.exists():
        if build.exists():
            raise RuntimeError('previous incomplete fixture build preserved at '+str(build))
        build.mkdir(parents=True, exist_ok=False)
        copies=[]
        for path in [source,*headers]:
            copied=build/path.name;copied.write_bytes(path.read_bytes());copies.append(copied)
        compile_command, link_command = fixture_commands(commands,obj,binary,build/source.name,build/'fixture.d')
        save(build/'started.json',dict(inputs=inputs,commands=[compile_command,link_command]))
        for i, command in enumerate([compile_command,link_command]):
            with (build/f'build_{i}.log').open('wb') as log:
                result = subprocess.run([LOADER,'--library-path',SYSTEM_LIBS,*command],env=compiler_env,
                cwd=build,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if result.returncode:
                save(build/'receipt.json',dict(status='FAILED_BUILD',inputs=inputs,exit_code=result.returncode,command=command))
                raise RuntimeError('build failed: '+str(build/f'build_{i}.log'))
        dependency_text=(build/'fixture.d').read_text().replace('\\\n',' ')
        dependencies=[Path(p) for p in dependency_text.split(':',1)[1].split()]
        save(build/'dependencies.json',{str(p):sha(p) for p in dependencies})
        save(build/'receipt.json',dict(status='BUILT',inputs=inputs,binary_sha256=sha(binary)))
    build_receipt=json.loads((build/'receipt.json').read_text())
    if build_receipt['inputs']!=inputs or build_receipt['binary_sha256']!=sha(binary):
        raise ValueError('fixture build identity changed')
    dependencies=json.loads((build/'dependencies.json').read_text())
    if any(sha(Path(p))!=expected for p,expected in dependencies.items()):
        raise ValueError('compiled include dependency changed')
    command=prefix+[str(binary),'--output='+str(output/'receiver.csv'),*arguments]
    if source_name=='wifi_threeway.cc' and not any(a.startswith('--flowMonitorOutput=') for a in arguments):
        command+=['--flowMonitorOutput='+str(output/'flow.xml')]
    native_started=time.perf_counter()
    with (output/'stdout.log').open('wb') as log:
        if stdin_path is not None:
            with stdin_path.open('rb') as policy_input:
                result=subprocess.run(command,cwd=output,env=env,stdin=policy_input,
                                      stdout=log,stderr=subprocess.STDOUT,timeout=timeout_seconds)
        else:
            result=subprocess.run(command,cwd=output,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,timeout=timeout_seconds)
    if stdin_path is not None and sha(stdin_path)!=receipt['stdin_sha256']:
        raise ValueError('copied policy input changed while native fixture ran')
    receipt.update(status='TERMINAL',exit_code=result.returncode,command=command,
                   build=str(build),binary_sha256=sha(binary),elapsed_seconds=time.perf_counter()-started,
                   native_elapsed_seconds=time.perf_counter()-native_started,
                   source_inputs=inputs,dependency_count=len(dependencies))
    # Persist the real terminal code before any later validation can fail.
    save(output/'receipt.json',receipt)
    if result.returncode:
        raise RuntimeError('native process exited with code '+str(result.returncode))
    if resource_profile is not None:
        rows=[json.loads(line.split(' ',1)[1]) for line in (output/'stdout.log').read_text().splitlines()
            if line.startswith('THREEWAY_RESOURCE ')]
        if len(rows)!=1 or rows[0]['cpus']!=[2]:
            raise ValueError('native packet process did not prove required affinity')
        receipt['native_resources']=rows[0]
    save(output/'receipt.json',receipt)
    save(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stdin-input',type=Path)
    parser.add_argument('--native-timeout',type=int,default=900)
    parser.add_argument('--campaign-resources',action='store_true')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    rest=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    try:
        result=run(args.source,args.output,rest,stdin_path=args.stdin_input,resource_profile=True if args.campaign_resources else None,timeout_seconds=args.native_timeout)
        print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'])))
    except Exception:
        if args.output.is_dir():
            error=traceback.format_exc()
            (args.output/'pipeline_error.txt').write_text(error,encoding='utf-8')
            receipt_path=args.output/'receipt.json'
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            receipt.update(status='FAILED_PIPELINE',terminal=True,error=error)
            receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            files={p.relative_to(args.output).as_posix():sha(p) for p in args.output.rglob('*')
                if p.is_file() and p!=args.output/'artifact_hashes.json'}
            (args.output/'artifact_hashes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
        raise
