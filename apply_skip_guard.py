"""Apply the optional non-finite-loss skip-guard to src/train_async.py.
Idempotent. Run from the repository root with the (oldpy) env active:
    python3 apply_skip_guard.py"""
try:
    p = 'src/train_async.py'
    s = open(p).read()
    if 'nonfinite_skips += 1' not in s:
        lines = [l for l in s.splitlines(keepends=True) if l.strip() != 'nonfinite_skips = 0']
        idx = next(i for i, l in enumerate(lines) if l.lstrip().startswith('def train_model('))
        j = idx + 1
        while not lines[j].rstrip().endswith(':'):
            j += 1
        indent = ' ' * (len(lines[j+1]) - len(lines[j+1].lstrip()))
        lines.insert(j + 1, indent + 'nonfinite_skips = 0\n')
        out = []; backw = 0
        for l in lines:
            if l.strip() == 'loss.backward()':
                ind = l[:len(l) - len(l.lstrip())]
                out.append(ind + 'if not torch.isfinite(loss):\n')
                out.append(ind + '    nonfinite_skips += 1\n')
                out.append(ind + '    if nonfinite_skips <= 5 or nonfinite_skips % 50 == 0:\n')
                out.append(ind + '        print(f"[skip] non-finite loss step (total skips: {nonfinite_skips})")\n')
                out.append(ind + '    continue\n')
                backw += 1
            out.append(l)
        open(p, 'w').writelines(out)
        import py_compile; py_compile.compile(p, doraise=True)
        print(f'skip-guard added ({backw} sites)')
    else:
        print('skip-guard already present')
except Exception as e:
    print('skip-guard SKIPPED (it is optional):', e)
