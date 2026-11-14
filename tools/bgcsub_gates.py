#!/usr/bin/env python3
"""Fast gate set for BGC tracer subcycling (bgc_physics_subcycle) on a small
MARBL case, as UCLA ROMS's pytest cases for subcycling run it.

usage:
  python3 tools/bgcsub_gates.py setup --nml CASE.nml --exe build_gpu/roms_gpu_cuda --dir DIR [--tiles2x2]
  python3 tools/bgcsub_gates.py run   --dir DIR [--mpiexec "mpiexec -n {n}"] [--timeout 600] [--only NAME,...]
  python3 tools/bgcsub_gates.py check --dir DIR [--ref REFDIR]

setup  writes one run directory per case under DIR (namelist.nml, output in
       DIR/<case>/out) and DIR/gates.json. CASE.nml is a GPU namelist of a small
       MARBL case with restart output; the gates were built on UCLA ROMS's
       bgc_real_marbl test domain (39 x 19 x 10, dt 20 s), where every run takes
       seconds on one GPU. setup sets the run length, tiling, restart period,
       marbl_timestep and the &bgc_settings keys; everything else (inputs,
       physics) comes from CASE.nml. Needs only the standard library.
run    runs the cases in order on the GPUs of the current node (inside a GPU
       job). {n} in --mpiexec is the rank count (tiles + 1 writer). A restart
       leg points its inifile at its source run's restart file just before it
       starts. roms.log in each run directory.
check  compares and prints one PASS/FAIL line per gate, then a summary.
       Needs numpy and netCDF4. --ref REFDIR also compares every case that
       completed in both directories bit for bit (all restart records and the
       BGCSUB lines): against a run of the base binary for the subcycling-off
       cases (G-OFF), or of an earlier commit for a change that must not move
       results.

Cases (k = window length in time steps; MARBL every 4 steps unless named):
  off, off2, offi    subcycling off (MARBL every 4, every 2; implicit free surface)
  k4, k2, k2m4, k4i  k = 4; k = 2 with MARBL every 2; k = 2 with MARBL every 4; k = 4 implicit
  k4full, k4leg      k = 4 for 24 steps against 12 steps restarted at step 12
  k4t22              k = 4 on 2 x 2 tiles (with --tiles2x2; needs 4 GPUs)
  w_short, w_frac, w_nodiv   rejected window lengths: dt/2, 1.5 dt, 3 dt with MARBL every 4
  rst_period         restart period 3 dt at k = 4
  st2, inwin         a stock run with a restart at step 2, then k = 4 started from it
Gates:
  G-RUN    each case that should finish writes its last restart record at its last step
  G-PHYS   physics (temp, salt, u, v, zeta, ubar, vbar) of every restart record identical
           to the subcycling-off twin; the BGC tracers differ
  G-BUD    BGCSUB lines (bgc_subcycle_diag = 1): one per window; budget <= 1e-14,
           shadow <= 1e-12, cont <= 1e-12 (UCLA ROMS: about 5e-15, 1e-14, 7e-15 on this case)
  G-RST    k4full and k4leg: every variable of the step-24 record identical; the leg's
           BGCSUB lines equal the last ones of k4full
  G-TILE   k4t22 and k4: every restart record identical (split free surface)
  G-MSG    the rejected settings stop at start-up with UCLA ROMS's message
"""
import json
import os
import re
import shlex
import subprocess
import sys
import time

PHYS = ['temp', 'salt', 'u', 'v', 'zeta', 'ubar', 'vbar']
SKIP = {'time_step', 'ocean_time', 'scrum_time', 'time'}
MSG_STOP = 'ERROR STOP bgc_subcycle'


# --------------------------------------------------------------------------- namelist text
def fval(v):
    if isinstance(v, bool):
        return '.true.' if v else '.false.'
    if isinstance(v, str):
        return "'" + v + "'"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def set_key(text, group, key, value):
    """Set group%key in namelist text (replacing the assignment and its
    continuation lines, or adding it); add the group if it is absent."""
    m = re.search(r'(?ims)^[ \t]*&' + group + r'\b[^\n]*\n(.*?)^[ \t]*/[ \t]*$', text)
    line = '    %s = %s\n' % (key, fval(value))
    if not m:
        return text.rstrip('\n') + '\n\n&%s\n%s/\n' % (group, line)
    body = m.group(1).splitlines(keepends=True)
    out, i, done = [], 0, False
    while i < len(body):
        if re.match(r'(?i)^\s*' + key + r'\s*=', body[i]):
            i += 1
            while i < len(body) and not re.match(r'^\s*\w+\s*(\(.*\))?\s*=', body[i]):
                i += 1
            out.append(line)
            done = True
            continue
        out.append(body[i])
        i += 1
    if not done:
        out.append(line)
    return text[:m.start(1)] + ''.join(out) + text[m.end(1):]


def get_key(text, group, key):
    m = re.search(r'(?ims)^[ \t]*&' + group + r'\b[^\n]*\n(.*?)^[ \t]*/[ \t]*$', text)
    if m:
        k = re.search(r'(?im)^\s*' + key + r'\s*=\s*([^\s,!]+)', m.group(1))
        if k:
            return k.group(1).strip("'\"")
    return None


# --------------------------------------------------------------------------- cases
def cases(dt, tiles2x2):
    """name -> (namelist edits, expectation); expectation 'ok' or a list of message parts."""
    sub = {'bgc_physics_subcycle': True, 'bgc_physics_timestep': 0.0, 'bgc_subcycle_diag': 1}
    off = {'bgc_physics_subcycle': False, 'bgc_physics_timestep': 0.0, 'bgc_subcycle_diag': 0}

    def c(nt=12, rst=4, marbl=4, bgc=None, imp=False, tiles=(1, 1), ini=None):
        return {'nt': nt, 'rst': rst * dt, 'marbl': marbl * dt, 'bgc': bgc or off, 'imp': imp,
                'tiles': tiles, 'ini': ini}
    rstmsg = ['restarts must fall on window boundaries', 'output_period_rst must be a multiple of']
    k = {
        'off': (c(), 'ok'),
        'off2': (c(marbl=2), 'ok'),
        'offi': (c(imp=True), 'ok'),
        'k4': (c(bgc=sub), 'ok'),
        'k2': (c(marbl=2, bgc=sub), 'ok'),
        'k2m4': (c(bgc=dict(sub, bgc_physics_timestep=2.0 * dt)), 'ok'),
        'k4i': (c(bgc=sub, imp=True), 'ok'),
        'k4full': (c(nt=24, rst=12, bgc=sub), 'ok'),
        'k4leg': (c(nt=12, rst=12, bgc=sub, ini=('k4full', 1)), 'ok'),
        'w_short': (c(bgc=dict(sub, bgc_physics_timestep=0.5 * dt)), ['is shorter than dt']),
        'w_frac': (c(bgc=dict(sub, bgc_physics_timestep=1.5 * dt)), ['is not a whole multiple of dt']),
        'w_nodiv': (c(bgc=dict(sub, bgc_physics_timestep=3.0 * dt)),
                    ['is not a whole multiple of bgc_physics_timestep']),
        'rst_period': (c(rst=3, bgc=sub), rstmsg),
        'st2': (c(nt=4, rst=2), 'ok'),
        'inwin': (c(nt=8, bgc=sub, ini=('st2', 1)), rstmsg + ['(time step 2 of the initial file']),
    }
    if tiles2x2:
        k['k4t22'] = (c(bgc=sub, tiles=(2, 2)), 'ok')
    return k


def setup(a):
    tpl = open(a['nml']).read()
    dt = float(get_key(tpl, 'time_stepping', 'dt'))
    exe = os.path.abspath(a['exe'])
    root = os.path.abspath(a['dir'])
    meta = {'exe': exe, 'nml': os.path.abspath(a['nml']), 'dt': dt, 'cases': {}}
    for name, (s, expect) in cases(dt, 'tiles2x2' in a).items():
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, 'out'), exist_ok=True)
        t = tpl
        edits = [('simulation_name_settings', 'output_root_name', os.path.join(d, 'out', 'output')),
                 ('time_stepping', 'ntimes', s['nt']), ('time_stepping', 'ninfo', 1),
                 ('param_settings', 'np_xi', s['tiles'][0]), ('param_settings', 'np_eta', s['tiles'][1]),
                 ('tile_settings', 'tile_balance', 'uniform'),
                 ('basic_output_settings', 'wrt_file_rst', True),
                 ('basic_output_settings', 'output_period_rst', float(s['rst'])),
                 ('basic_output_settings', 'wrt_file_his', False),
                 ('basic_output_settings', 'wrt_file_avg', False),
                 ('marbl_biogeochemistry_settings', 'marbl_timestep', float(s['marbl'])),
                 ('implicit_fs', 'fs_mode', 'implicit' if s['imp'] else 'split')]
        edits += [('bgc_settings', kk, v) for kk, v in s['bgc'].items()]
        for g, kk, v in edits:
            t = set_key(t, g, kk, v)
        open(os.path.join(d, 'namelist.nml'), 'w').write(t)
        meta['cases'][name] = {'nt': s['nt'], 'tiles': s['tiles'], 'ini': s['ini'], 'expect': expect,
                               'k': (0 if not s['bgc']['bgc_physics_subcycle'] else
                                     int(round((s['bgc']['bgc_physics_timestep'] or s['marbl']) / dt)))}
    json.dump(meta, open(os.path.join(root, 'gates.json'), 'w'), indent=1)
    print('setup: %d cases in %s (dt %g s, binary %s)' % (len(meta['cases']), root, dt, exe))


# --------------------------------------------------------------------------- run
def rst_files(d):
    o = os.path.join(d, 'out')
    if not os.path.isdir(o):
        return []
    return sorted(os.path.join(o, f) for f in os.listdir(o) if re.match(r'.*_rst\.\d+\.nc$', f))


def run(a):
    root = os.path.abspath(a['dir'])
    meta = json.load(open(os.path.join(root, 'gates.json')))
    tmpl = a.get('mpiexec', 'mpiexec -n {n}')
    only = set(a['only'].split(',')) if 'only' in a else None
    for name, c in meta['cases'].items():
        if only and name not in only:
            continue
        d = os.path.join(root, name)
        for f in rst_files(d):
            os.remove(f)
        nml = os.path.join(d, 'namelist.nml')
        if c['ini']:
            src = rst_files(os.path.join(root, c['ini'][0]))
            if len(src) < c['ini'][1]:
                print('RUN %-10s skipped: no restart record %d in %s' % (name, c['ini'][1], c['ini'][0]))
                continue
            t = set_key(open(nml).read(), 'initial_conditions', 'inifile', src[c['ini'][1] - 1])
            open(nml, 'w').write(t)
        n = c['tiles'][0] * c['tiles'][1] + 1
        cmd = shlex.split(tmpl.format(n=n)) + [meta['exe'], 'run', nml]
        t0 = time.time()
        with open(os.path.join(d, 'roms.log'), 'w') as log:
            try:
                rc = subprocess.call(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=float(a.get('timeout', 600)))
            except subprocess.TimeoutExpired:
                rc = 'timeout'
        print('RUN %-10s rc %s, %.1f s' % (name, rc, time.time() - t0), flush=True)


# --------------------------------------------------------------------------- check
def bgcsub_lines(d):
    try:
        t = open(os.path.join(d, 'roms.log'), errors='ignore').read()
    except FileNotFoundError:
        return []
    return [ln.strip() for ln in t.splitlines() if re.match(r'^\s*BGCSUB\s+\d', ln)]


def check(a):
    import numpy as np
    import netCDF4 as nc
    root = os.path.abspath(a['dir'])
    meta = json.load(open(os.path.join(root, 'gates.json')))
    C = meta['cases']
    res = []

    def gate(name, ok, info):
        res.append(ok)
        print('%s  %-8s %s' % ('PASS' if ok else 'FAIL', name, info))

    def same_vars(fa, fb, names=None, rec=None):
        """variables of two restart files: (number identical, list differing)"""
        with nc.Dataset(fa) as A, nc.Dataset(fb) as B:
            vs = [v for v in A.variables if v in B.variables and v not in SKIP]
            if names is not None:
                vs = [v for v in vs if v in names]
            nsame, bad = 0, []
            for v in vs:
                x, y = np.asarray(A[v][:]), np.asarray(B[v][:])
                if x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
                    nsame += 1
                else:
                    bad.append(v)
            return nsame, bad

    def tstep(f):
        # the last record's step (time_step(:, 1) is the step count of each record)
        with nc.Dataset(f) as A:
            return int(np.asarray(A['time_step'][:])[..., 0].max()) if 'time_step' in A.variables else -1

    d = lambda n: os.path.join(root, n)
    done = {}
    for n, c in C.items():
        log = open(os.path.join(d(n), 'roms.log'), errors='ignore').read() if os.path.exists(
            os.path.join(d(n), 'roms.log')) else ''
        if c['expect'] == 'ok':
            f = rst_files(d(n))
            # a restart leg's steps count on from its initial file
            want = c['nt']
            if c['ini']:
                src = rst_files(d(c['ini'][0]))
                want += tstep(src[c['ini'][1] - 1]) if len(src) >= c['ini'][1] else 0
            last = tstep(f[-1]) if f else None
            ok = bool(f) and 'ERROR' not in log and last == want
            done[n] = ok
            gate('G-RUN', ok, '%-10s %d restart file(s), last record step %s (run of %d steps)'
                 % (n, len(f), last, want))
        else:
            ok = MSG_STOP in log and all(p in log for p in c['expect'])
            ls = log.splitlines()
            m = [ls[i + 1].strip() for i, ln in enumerate(ls[:-1]) if 'ERROR [global]' in ln]
            gate('G-MSG', ok, '%-10s %s' % (n, (m[0] if m else 'no message')[:160]))
    for k, o in (('k4', 'off'), ('k2', 'off2'), ('k2m4', 'off'), ('k4i', 'offi')):
        if not (done.get(k) and done.get(o)):
            gate('G-PHYS', False, '%s vs %s: a run did not finish' % (k, o))
            continue
        fk, fo = rst_files(d(k)), rst_files(d(o))
        ok, info = len(fk) == len(fo), []
        ndiff_bgc = 0
        for a_, b_ in zip(fk, fo):
            ns, bad = same_vars(a_, b_, PHYS)
            ok = ok and not bad
            with nc.Dataset(a_) as A:
                bgc = [v for v in A.variables if v not in PHYS and v not in SKIP and A[v].ndim == 4]
            ns2, bad2 = same_vars(a_, b_, bgc)
            ndiff_bgc += len(bad2)
            info.append('%d/%d' % (ns, ns + len(bad)))
        ok = ok and ndiff_bgc > 0
        gate('G-PHYS', ok, '%-5s vs %-4s physics identical in records: %s; BGC variables differing: %d'
             % (k, o, ' '.join(info), ndiff_bgc))
    for n, c in C.items():
        if not c['k'] or not done.get(n):
            continue
        L = bgcsub_lines(d(n))
        num = [re.findall(r'(cont|shadow|budget)\s+(\S+)', ln) for ln in L]
        mx = {q: max([float(v) for x in num for (k_, v) in x if k_ == q] or [float('nan')])
              for q in ('cont', 'shadow', 'budget')}
        ms = max([int(re.search(r'\sm\s+(\d+)', ln).group(1)) for ln in L] or [0])
        nwin = c['nt'] // c['k']
        ok = len(L) == nwin and mx['budget'] <= 1e-14 and mx['shadow'] <= 1e-12 and mx['cont'] <= 1e-12
        gate('G-BUD', ok, '%-10s windows %d/%d, max m %d, budget %.2e, shadow %.2e, cont %.2e'
             % (n, len(L), nwin, ms, mx['budget'], mx['shadow'], mx['cont']))
    if done.get('k4full') and done.get('k4leg'):
        ns, bad = same_vars(rst_files(d('k4full'))[-1], rst_files(d('k4leg'))[-1])
        Lf, Ll = bgcsub_lines(d('k4full')), bgcsub_lines(d('k4leg'))
        okl = len(Ll) > 0 and Lf[-len(Ll):] == Ll
        gate('G-RST', not bad and okl, 'k4full vs k4leg at step 24: %d variables identical, differing: %s; '
             'leg BGCSUB lines (%d) equal the last of k4full: %s' % (ns, ','.join(bad) or 'none', len(Ll), okl))
    else:
        gate('G-RST', False, 'k4full or k4leg did not finish')
    if 'k4t22' in C:
        if done.get('k4t22') and done.get('k4'):
            pairs = list(zip(rst_files(d('k4')), rst_files(d('k4t22'))))
            bad = [b for f1, f2 in pairs for b in same_vars(f1, f2)[1]]
            gate('G-TILE', not bad and len(pairs) > 0, 'k4t22 vs k4: %d records, differing: %s'
                 % (len(pairs), ','.join(sorted(set(bad))) or 'none'))
        else:
            gate('G-TILE', False, 'k4 or k4t22 did not finish')
    if 'ref' in a:
        ref = os.path.abspath(a['ref'])
        for n in C:
            if not done.get(n) or not rst_files(os.path.join(ref, n)):
                continue
            fa, fb = rst_files(os.path.join(ref, n)), rst_files(d(n))
            bad = [b for f1, f2 in zip(fa, fb) for b in same_vars(f1, f2)[1]]
            okl = bgcsub_lines(os.path.join(ref, n)) == bgcsub_lines(d(n))
            gate('G-REF', not bad and len(fa) == len(fb) and okl,
                 '%-10s vs %s: %d/%d records, differing: %s; BGCSUB lines identical: %s'
                 % (n, ref, len(fb), len(fa), ','.join(sorted(set(bad))) or 'none', okl))
    print('%s: %d of %d gates pass' % ('ALL PASS' if all(res) else 'FAILURES', sum(res), len(res)))
    return 0 if all(res) else 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('setup', 'run', 'check'):
        print(__doc__)
        return 2
    a, args = {}, sys.argv[2:]
    i = 0
    while i < len(args):
        k = args[i].lstrip('-')
        if k == 'tiles2x2':
            a[k] = True
            i += 1
        else:
            a[k] = args[i + 1]
            i += 2
    if sys.argv[1] == 'setup':
        setup(a)
    elif sys.argv[1] == 'run':
        run(a)
    else:
        return check(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
