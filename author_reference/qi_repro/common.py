from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd


def dump_json(path, obj):
    """Reject NaN/Infinity so logs are valid, portable JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def numeric_list(text, comma=True):
    # literal_eval只解析字面量；不使用eval执行数据中的字符串。
    if comma:
        result = np.asarray(ast.literal_eval(str(text)), dtype=float)
    else:
        result = np.asarray([float(x) for x in str(text).strip('[] ').split()], dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError('Invalid numeric sequence')
    return result


def save_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def require_files(root):
    names = ['df_test.csv', 'control_demand_vlt.csv', 'order_e2e_pre_500.csv',
             'order_e2e_post_500.csv', 'order_jd_pre_500.csv', 'order_jd_post_500.csv']
    missing = [n for n in names if not (Path(root) / n).is_file()]
    if missing:
        raise FileNotFoundError('Required author data missing: ' + ', '.join(missing))


def audit_inputs(root, output):
    import hashlib
    rows=[]
    for p in sorted(Path(root).rglob('*.csv')):
        df=pd.read_csv(p)
        rows.append({'file':p.relative_to(root).as_posix(), 'rows':len(df),
                     'columns':list(df.columns), 'missing_cells':int(df.isna().sum().sum()),
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dump_json(output, rows)
    return rows
