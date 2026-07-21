"""Visual identity for qwen-voice-studio — a warm analog-studio console.

Deep warm ink, an amber "signal" accent (VU-meter / tube warmth), coral for hot
states; Bricolage Grotesque display over IBM Plex Sans/Mono. The signature is the
live signal-bar motif in the header and amber waveforms on every output.
"""
from __future__ import annotations

import gradio as gr

# ---- palette -----------------------------------------------------------------
INK = "#141009"        # warm near-black base
PANEL = "#1E190F"      # raised panel
PANEL_2 = "#282013"    # control surface
LINE = "rgba(240, 200, 120, 0.14)"
TEXT = "#F1E8D6"       # warm off-white
MUTED = "#A9997C"      # warm taupe
SIGNAL = "#F0A93B"     # amber — the accent
SIGNAL_HOT = "#FF6B4A" # coral — hot / recording / peaks


def studio_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill=INK,
        body_text_color=TEXT,
        body_text_color_subdued=MUTED,
        background_fill_primary=PANEL,
        background_fill_secondary=INK,
        block_background_fill=PANEL,
        block_border_color=LINE,
        block_border_width="1px",
        block_label_text_color=MUTED,
        block_title_text_color=TEXT,
        border_color_primary=LINE,
        input_background_fill=PANEL_2,
        input_border_color=LINE,
        button_primary_background_fill=SIGNAL,
        button_primary_background_fill_hover="#FFC15A",
        button_primary_text_color=INK,
        button_secondary_background_fill="transparent",
        button_secondary_text_color=TEXT,
        button_secondary_border_color=LINE,
        slider_color=SIGNAL,
        color_accent=SIGNAL,
        color_accent_soft=PANEL_2,
    )


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --ink: {INK}; --panel: {PANEL}; --panel-2: {PANEL_2}; --line: {LINE};
  --text: {TEXT}; --muted: {MUTED}; --signal: {SIGNAL}; --hot: {SIGNAL_HOT};
}}

.gradio-container {{ max-width: 1180px !important; margin: 0 auto !important; }}
body, .gradio-container {{
  background:
    radial-gradient(1200px 500px at 80% -10%, rgba(240,169,59,0.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(255,107,74,0.06), transparent 55%),
    var(--ink) !important;
}}

/* ---- header / wordmark ---- */
.qvs-header {{ padding: 26px 4px 10px; }}
.qvs-brand {{ display:flex; align-items:center; gap:18px; }}
.qvs-wordmark {{
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800;
  font-size: clamp(30px, 4.4vw, 52px); line-height:0.95; letter-spacing:-0.02em;
  color: var(--text); margin:0;
}}
.qvs-wordmark .dot {{ color: var(--signal); }}
.qvs-tag {{
  font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.18em;
  text-transform:uppercase; color: var(--muted); margin-top:8px;
}}
/* live signal bars — the signature */
.qvs-signal {{ display:flex; align-items:flex-end; gap:3px; height:34px; }}
.qvs-signal i {{
  width:4px; background:linear-gradient(var(--signal), var(--hot)); border-radius:2px;
  animation: qvs-bounce 1.1s ease-in-out infinite; opacity:0.9;
}}
.qvs-signal i:nth-child(1){{height:40%;animation-delay:-.9s}} .qvs-signal i:nth-child(2){{height:75%;animation-delay:-.7s}}
.qvs-signal i:nth-child(3){{height:55%;animation-delay:-.5s}} .qvs-signal i:nth-child(4){{height:95%;animation-delay:-.3s}}
.qvs-signal i:nth-child(5){{height:60%;animation-delay:-.15s}} .qvs-signal i:nth-child(6){{height:85%;animation-delay:-.55s}}
.qvs-signal i:nth-child(7){{height:45%;animation-delay:-.35s}} .qvs-signal i:nth-child(8){{height:70%;animation-delay:-.8s}}
@keyframes qvs-bounce {{ 0%,100%{{transform:scaleY(0.35)}} 50%{{transform:scaleY(1)}} }}

/* console meter readout */
.qvs-meter, .qvs-meter * {{ font-family:'IBM Plex Mono', monospace !important; }}
.qvs-meter {{
  font-size:11.5px; letter-spacing:0.04em; color:var(--muted);
  border:1px solid var(--line); border-radius:6px; padding:8px 12px; background:rgba(0,0,0,0.25);
}}
.qvs-meter b {{ color: var(--signal); font-weight:500; }}

/* eyebrow labels */
.qvs-eyebrow {{
  font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.16em;
  text-transform:uppercase; color:var(--muted); margin:2px 0 2px;
}}
.qvs-eyebrow .num {{ color:var(--signal); }}

/* tabs as console channels */
.tab-nav {{ border-bottom:1px solid var(--line) !important; gap:2px; }}
.tab-nav button {{
  font-family:'IBM Plex Mono', monospace !important; text-transform:uppercase;
  letter-spacing:0.08em; font-size:12.5px !important; color:var(--muted) !important;
  border:none !important; border-radius:0 !important; padding:12px 16px !important;
}}
.tab-nav button.selected {{
  color:var(--text) !important; box-shadow: inset 0 -2px 0 var(--signal);
  background:transparent !important;
}}

/* buttons */
button.primary, .qvs-generate button {{
  font-family:'IBM Plex Mono', monospace !important; text-transform:uppercase;
  letter-spacing:0.1em; font-weight:500 !important;
}}
.qvs-generate button {{ box-shadow: 0 6px 22px -8px rgba(240,169,59,0.55); }}

/* panels */
.block, .form {{ border-color: var(--line) !important; }}
.qvs-panel {{ background: var(--panel); border:1px solid var(--line); border-radius:10px; padding:6px 14px 14px; }}

/* audio waveform -> amber */
.gradio-container [data-testid="waveform"] ::selection {{ background: var(--signal); }}
:root {{ --wave-color: {SIGNAL}; --wave-progress-color: {SIGNAL_HOT}; }}

/* generating status */
.qvs-status {{ font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.05em; color:var(--muted); }}
.qvs-status.on {{ color: var(--hot); }}
.qvs-status.on::before {{ content:'● '; animation: qvs-blink 1s steps(2) infinite; }}
@keyframes qvs-blink {{ 50% {{ opacity:0.25; }} }}

footer {{ display:none !important; }}
.qvs-foot {{
  font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--muted);
  text-align:center; padding:26px 0 12px; letter-spacing:0.04em; border-top:1px solid var(--line); margin-top:22px;
}}
.qvs-foot a {{ color: var(--signal); text-decoration:none; }}

@media (prefers-reduced-motion: reduce) {{
  .qvs-signal i {{ animation:none; }} .qvs-status.on::before {{ animation:none; }}
}}
@media (max-width: 720px) {{
  .qvs-brand {{ flex-direction:column; align-items:flex-start; gap:10px; }}
}}
"""


def header_html() -> str:
    bars = "".join("<i></i>" for _ in range(8))
    return f"""
    <div class="qvs-header">
      <div class="qvs-brand">
        <div class="qvs-signal">{bars}</div>
        <div>
          <h1 class="qvs-wordmark">Qwen Voice Studio<span class="dot">.</span></h1>
          <div class="qvs-tag">clone · design · direct — powered by Qwen3-TTS 1.7B</div>
        </div>
      </div>
    </div>
    """


def footer_html() -> str:
    return (
        '<div class="qvs-foot">Qwen Voice Studio — open source · Apache-2.0 · '
        'built on <a href="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base">Qwen3-TTS</a>. '
        'Generated audio is synthetic; use responsibly and only with consent.</div>'
    )
