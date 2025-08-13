
import os, io, time, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE = os.path.join(os.path.dirname(__file__), "qsw_go_nogo_v2")
if not os.path.isdir(BASE):
    st.error("Folder 'qsw_go_nogo_v2' not found. Add your project files next to this app.")
    st.stop()

import sys
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from importlib import reload
import qrng_client_v2, agent_workspace_hebbs, tasks_impasse
reload(qrng_client_v2); reload(agent_workspace_hebbs); reload(tasks_impasse)
from qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity
from sklearn.metrics import mutual_info_score

st.sidebar.title("QSW v2 — Controls")
trials = st.sidebar.slider("Trials per condition", 200, 2000, 1000, 100)
phase_len = st.sidebar.slider("Phase length", 8, 20, 12, 1)
max_steps = st.sidebar.slider("Max steps per trial", 20, 60, 30, 1)
st.sidebar.write("---")
cond_quantum = st.sidebar.checkbox("Quantum (ANU)", True)
cond_pseudo  = st.sidebar.checkbox("Pseudo", True)
cond_det     = st.sidebar.checkbox("Deterministic", True)
st.sidebar.write("---")
min_qratio   = st.sidebar.slider("Min quantum ratio", 0.5, 1.0, 0.90, 0.01)
hc_total     = st.sidebar.slider("Health-check draws", 2000, 20000, 8000, 1000)
batch_small  = st.sidebar.slider("Health-check batch", 256, 4096, 1024, 256)
batch_run    = st.sidebar.slider("Runtime batch size", 512, 20000, 4096, 512)
max_retries  = st.sidebar.slider("Max retries/refill", 3, 12, 8, 1)
backoff      = st.sidebar.slider("Backoff factor", 0.5, 2.0, 1.2, 0.1)

st.title("QSW v2 — Impasse Escape (ANU QRNG, Streamlit)")

def save_bar(df, col, title, ylabel, path):
    fig, ax = plt.subplots()
    ax.bar(df["condition"], df[col])
    ax.set_title(title); ax.set_xlabel("Condition"); ax.set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def health_check(hc_total, batch_small, max_retries, backoff, min_qratio):
    log = io.StringIO()
    c = QRNGClient('quantum', batch_size=batch_small, max_retries=max_retries, backoff=backoff)
    draws = 0
    bar = st.progress(0, text="Health-check: contacting ANU…")
    while draws < hc_total:
        if not getattr(c, "_buf", []):
            bq = getattr(c, "true_quantum_count", 0)
            bf = getattr(c, "fallback_count", 0)
            try: c._refill(batch_small)
            except Exception: time.sleep(0.3)
            got_q = getattr(c, "true_quantum_count", 0) - bq
            got_f = getattr(c, "fallback_count", 0) - bf
            ratio = getattr(c, "true_quantum_count", 0) / max(1, getattr(c, "true_quantum_count", 0) + getattr(c, "fallback_count", 0))
            print(f"refill: +{got_q} quantum, +{got_f} fallback | cumulative ratio={ratio:.3f}", file=log)
        take = min(len(getattr(c, "_buf", [])), max(1, batch_small//2))
        for _ in range(take): c._buf.pop()
        draws += take
        bar.progress(min(100, int(100*draws/float(hc_total))), text=f"Health-check: {draws}/{hc_total}")
        if draws % (batch_small*4) == 0: time.sleep(0.1)
    q = getattr(c, "true_quantum_count", 0); f = getattr(c, "fallback_count", 0)
    ratio = q / max(1, q+f)
    print(f"COMPLETE → true_quantum={q}, fallback={f}, quantum_ratio={ratio:.3f}, total_drawn={draws}", file=log)
    return (ratio >= min_qratio), ratio, log.getvalue()

def run_condition(n_trials, condition, seed, phase_len, max_steps, batch_run, max_retries, backoff):
    import numpy as np
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []
    if condition == 'quantum':
        src = QRNGClient('quantum', batch_size=batch_run, max_retries=max_retries, backoff=backoff)
        try: src._refill(batch_run)
        except: pass
    elif condition == 'pseudo':
        src = QRNGClient('pseudo')
    else:
        src = QRNGClient('deterministic')

    def next_gate():
        if condition == 'quantum' and not getattr(src, '_buf', []): src._refill(batch_run)
        v = (src._buf.pop() if getattr(src, '_buf', []) else src.next()) * 1.5
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    prog = st.progress(0, text=f"{condition}: starting…")
    for i in range(1, n_trials+1):
        agent = WorkspaceAgentHebb(seed=int(rng.integers(0, 1_000_000_000)))
        task  = ImpasseEscapeTask(seed=int(rng.integers(0, 1_000_000_000)), phase_len=phase_len)
        actions, time_to_escape, success = [], None, False
        for t in range(max_steps):
            ctx = tuple(actions[-3:]) if actions else None
            g = next_gate()
            if t < 4: early_gates.append(g)
            a, probs = agent.step(g, ctx)
            actions.append(a)
            if len(actions) >= task.phase_len:
                w = actions[-task.phase_len:]
                if time_to_escape is None and w[:3]==task.prefix and w!=task.decoy and w==task.goal:
                    time_to_escape = t+1
                if w == task.goal:
                    success = True; break
        if success: successes += 1
        times.append(time_to_escape if time_to_escape is not None else max_steps)
        sequences.append(tuple(actions))
        late_outcomes.append(1 if success else 0)
        if i % max(1, n_trials//20) == 0 or i == n_trials:
            prog.progress(int(100*i/n_trials), text=f"{condition}: {i}/{n_trials}")

    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates); bins = np.linspace(0,1,6)
    dig = np.digitize(eg, bins)-1 if len(eg) else np.array([])
    lo = np.repeat(late_outcomes, 4)
    L = min(len(dig), len(lo))
    try: mi = float(mutual_info_score(dig[:L], lo[:L])) if L>1 else 0.0
    except Exception: mi = 0.0

    q_ratio = None
    if condition == 'quantum':
        q = getattr(src, 'true_quantum_count', 0); f = getattr(src, 'fallback_count', 0)
        q_ratio = q / max(1, q+f)

    return {
        'condition': condition,
        'n_trials': n_trials,
        'success_rate': successes / max(1,n_trials),
        'avg_time_to_escape': float(np.mean(times)),
        'diversity_bits': H,
        'unique_seq': unique,
        'early_gate_success_MI': mi,
        'quantum_ratio': q_ratio
    }

def build_outputs(df, status):
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(os.path.dirname(__file__), 'QSW_runs')
    out_dir = os.path.join(out_root, f"{stamp}_v2_streamlit_{status}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    p1 = os.path.join(out_dir, 'success_rate.png')
    p2 = os.path.join(out_dir, 'time_to_escape.png')
    p3 = os.path.join(out_dir, 'diversity_bits.png')
    p4 = os.path.join(out_dir, 'mi.png')
    save_bar(df, 'success_rate', 'Success Rate — Impasse Escape (v2)', 'Success Rate', p1)
    save_bar(df, 'avg_time_to_escape', 'Avg Time to Escape (lower=better)', 'Avg Time to Escape', p2)
    save_bar(df, 'diversity_bits', 'Diversity (bits)', 'Diversity (bits)', p3)
    save_bar(df, 'early_gate_success_MI', 'Early Gate ↔ Success MI (nats)', 'MI (nats)', p4)

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    pdf_path = os.path.join(out_dir, f"QSW_Run_Summary_v2_{status}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter); w, h = letter
    c.setFont("Helvetica-Bold", 16); c.drawCentredString(w/2, h-40, "QSW v2 — Impasse Escape (ANU QRNG)")
    c.setFont("Helvetica", 10)
    c.drawString(40, h-60, f"UTC Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if 'quantum' in set(df['condition']):
        qr_final = float(df.loc[df['condition']=='quantum','quantum_ratio'].fillna(0).values[0])
        c.drawString(40, h-75, f"Quantum ratio (run): {qr_final:.3f}")
    c.setFont("Helvetica-Bold", 12); c.drawString(40, h-100, "Aggregate Metrics")
    c.setFont("Helvetica", 10); y = h-118
    for _, row in df.iterrows():
        line = f"{row['condition']:<12}  success={row['success_rate']:.3f}  avg_escape={row['avg_time_to_escape']:.2f}  H={row['diversity_bits']:.3f}  MI={row['early_gate_success_MI']:.4f}  uniq={int(row['unique_seq'])}  q_ratio={row.get('quantum_ratio', float('nan'))}"
        c.drawString(40, y, line[:110]); y -= 14
    for img in [p1,p2,p3,p4]:
        c.showPage(); c.drawImage(ImageReader(img), 60, 200, width=w-120, height=300, preserveAspectRatio=True, mask='auto')
    c.save()

    zip_path = f"{out_dir}.zip"
    shutil.make_archive(out_dir, 'zip', out_dir)
    return out_dir, csv_path, pdf_path, zip_path, [p1,p2,p3,p4]

st.write("1) Optional: Health-check ANU")
do_hc = st.checkbox("Run health-check before experiment", True)
if do_hc and st.button("Run Health-check now"):
    with st.spinner("Health-checking ANU…"):
        ok, ratio, log = health_check(hc_total, batch_small, max_retries, backoff, min_qratio)
    st.text_area("Health-check log", log, height=160)
    st.write(f"Quantum ratio: **{ratio:.3f}**")
    if not ok:
        st.warning(f"Health-check ratio below threshold {min_qratio:.2f}. Try again later.")
        st.stop()

st.write("2) Run experiment")
if st.button("Run QSW v2"):
    conditions = []
    if cond_quantum: conditions.append('quantum')
    if cond_pseudo:  conditions.append('pseudo')
    if cond_det:     conditions.append('deterministic')
    if not conditions:
        st.error("Select at least one condition."); st.stop()

    results = []
    for cond in conditions:
        st.write(f"### Running: {cond}")
        res = run_condition(trials, cond, seed=0, phase_len=phase_len, max_steps=max_steps,
                            batch_run=batch_run, max_retries=max_retries, backoff=backoff)
        results.append(res)

    df = pd.DataFrame(results)
    st.write("### Results Summary"); st.dataframe(df)

    status = "OK"
    if 'quantum' in set(df['condition']):
        qr_final = float(df.loc[df['condition']=='quantum','quantum_ratio'].fillna(0).values[0])
        status = "OK" if qr_final >= min_qratio else "LOW_QRATIO"

    out_dir, csv_path, pdf_path, zip_path, imgs = build_outputs(df, status)

    st.write("### Plots")
    c1, c2 = st.columns(2)
    c1.image(imgs[0], caption="Success Rate"); c2.image(imgs[1], caption="Avg Time to Escape")
    c3, c4 = st.columns(2)
    c3.image(imgs[2], caption="Diversity (bits)"); c4.image(imgs[3], caption="Early Gate ↔ Success MI")

    st.write("### Downloads")
    with open(csv_path, "rb") as f: st.download_button("Download results_summary.csv", f, file_name=os.path.basename(csv_path), mime="text/csv")
    with open(pdf_path, "rb") as f: st.download_button("Download QSW_Run_Summary.pdf", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
    with open(zip_path, "rb") as f: st.download_button("Download entire run (.zip)", f, file_name=os.path.basename(zip_path), mime="application/zip")
