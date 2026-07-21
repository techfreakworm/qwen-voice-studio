"""Qwen Voice Studio — Gradio app (runs on local MPS and Hugging Face ZeroGPU).

Five channels over the three Qwen3-TTS-12Hz-1.7B checkpoints: Clone, Preset
Voices, Voice Design, LoRA Lab, and a Voice Library that ties them together.
"""
from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np

from qvs import audio as qaudio
from qvs import config, engine, voices
from qvs.device import get_attn_impl, target_device
from qvs.lora import LoraManager, load_speaker_embedding
from qvs.memory import snapshot
from qvs.registry import ModelRegistry
from qvs.ui import theme

REG = ModelRegistry()
LORA = LoraManager()

LANG_CHOICES = list(config.LANGUAGES.keys())
SPEAKER_CHOICES = [(f"{s.display} — {s.description.rstrip('.')} ({s.language})", s.key) for s in config.SPEAKERS]

if os.environ.get("QVS_MEMGUARD"):
    from qvs.memory import MemoryGuard
    MemoryGuard(hard_gb=float(os.environ.get("QVS_MEMGUARD", "80"))).start()


# ---- helpers -----------------------------------------------------------------
def meter_html() -> str:
    snap = snapshot()
    lora = f' · LoRA <b>{LORA.state.source.split("/")[-1]}</b>' if LORA.state else ""
    return (
        f'<div class="qvs-meter">DEVICE <b>{target_device()}</b> · DTYPE <b>bf16</b> · '
        f'ATTN <b>{get_attn_impl()}</b> · MEM <b>{snap.committed:.0f}</b>/{snap.total:.0f} GB{lora}</div>'
    )


def gp(a) -> engine.GenParams:
    """Build GenParams from the 9 advanced-control values (positional)."""
    return engine.GenParams(
        temperature=float(a[0]), top_p=float(a[1]), top_k=int(a[2]), repetition_penalty=float(a[3]),
        subtalker_temperature=float(a[4]), subtalker_top_p=float(a[5]), subtalker_top_k=int(a[6]),
        max_new_tokens=int(a[7]), seed=int(a[8]),
    )


def advanced_controls():
    """Shared 'Advanced' rack. Returns the 9 components in GenParams order."""
    d = config.GEN_DEFAULTS
    with gr.Accordion("Advanced — sampling & sub-talker", open=False):
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, d.temperature, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, d.top_p, step=0.05, label="Top-p")
            top_k = gr.Slider(0, 100, d.top_k, step=1, label="Top-k")
            repetition_penalty = gr.Slider(1.0, 2.0, d.repetition_penalty, step=0.01, label="Repetition penalty")
        with gr.Row():
            st_temp = gr.Slider(0.0, 1.5, d.subtalker_temperature, step=0.05, label="Sub-talker temp")
            st_top_p = gr.Slider(0.0, 1.0, d.subtalker_top_p, step=0.05, label="Sub-talker top-p")
            st_top_k = gr.Slider(0, 100, d.subtalker_top_k, step=1, label="Sub-talker top-k")
        with gr.Row():
            max_new = gr.Slider(128, 4096, d.max_new_tokens, step=64, label="Max new tokens")
            seed = gr.Number(d.seed, precision=0, label="Seed (-1 = random)")
    return [temperature, top_p, top_k, repetition_penalty, st_temp, st_top_p, st_top_k, max_new, seed]


def status_line(msg: str, hot: bool = False) -> str:
    return f'<div class="qvs-status {"on" if hot else ""}">{msg}</div>'


def _done(t0: float, wav) -> str:
    return status_line(f"done · {len(wav)/config.OUTPUT_SAMPLE_RATE:.1f}s audio in {time.time()-t0:.1f}s")


# ---- callbacks ---------------------------------------------------------------
def do_preset(text, speaker, instruct, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter some text to speak.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("custom_voice")
    wav, sr = engine.synth_custom_voice(model, text.strip(), speaker, instruct, config.LANGUAGES[language], gp(adv), bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


def do_design(text, instruct, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter some text to speak.", hot=True), meter_html()
    if not (instruct or "").strip():
        return None, status_line("Describe the voice you want to design.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("voice_design")
    wav, sr = engine.synth_voice_design(model, text.strip(), instruct.strip(), config.LANGUAGES[language], gp(adv), bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


def do_clone(ref_audio, ref_text, xvec, text, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter text to synthesize in the cloned voice.", hot=True), meter_html()
    ref = qaudio.ref_from_gradio(ref_audio)
    if ref is None:
        return None, status_line("Upload or record reference audio first.", hot=True), meter_html()
    if not xvec and not (ref_text or "").strip():
        return None, status_line("Add the reference transcript, or enable x-vector-only mode.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("base")
    wav, sr = engine.synth_clone(model, text.strip(), config.LANGUAGES[language], gp(adv),
                                 ref_audio=ref, ref_text=(ref_text or None), x_vector_only=bool(xvec), longform=bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


def do_save_voice(name, ref_audio, ref_text, xvec):
    if not (name or "").strip():
        return gr.update(), status_line("Give the voice a name.", hot=True)
    ref = qaudio.ref_from_gradio(ref_audio)
    if ref is None:
        return gr.update(), status_line("Upload reference audio to save.", hot=True)
    if not xvec and not (ref_text or "").strip():
        return gr.update(), status_line("Reference transcript required (or enable x-vector-only).", hot=True)
    model = REG.to_device("base")
    voices.save_voice(model, name.strip(), ref, (ref_text or None), bool(xvec))
    return gr.update(choices=voices.list_voices(), value=name.strip()), status_line(f'saved voice "{name.strip()}"')


def do_library_gen(voice_name, text, language, longform, *adv):
    if not voice_name:
        return None, status_line("Pick a saved voice.", hot=True), meter_html()
    if not (text or "").strip():
        return None, status_line("Enter text to speak.", hot=True), meter_html()
    t0 = time.time()
    items = voices.load_voice(voice_name)
    model = REG.to_device("base")
    wav, sr = engine.synth_clone(model, text.strip(), config.LANGUAGES[language], gp(adv),
                                 voice_clone_prompt=items, longform=bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


def do_apply_lora(source):
    if not (source or "").strip():
        return status_line("Enter a Hugging Face repo id or local path.", hot=True), meter_html()
    try:
        st = LORA.apply(REG.to_device("base"), source.strip(), merge=True)
    except Exception as e:  # surface load errors in the UI voice
        return status_line(f"Couldn't load adapter: {type(e).__name__}: {e}", hot=True), meter_html()
    emb = " · ships a speaker voice" if st.has_speaker_embedding else ""
    return status_line(f"applied {st.n_modules} LoRA modules from {source.split('/')[-1]}{emb}"), meter_html()


def do_remove_lora():
    if not LORA.state:
        return status_line("No adapter is applied."), meter_html()
    REG.reload("base")
    LORA.clear()
    return status_line("removed adapter — Base restored"), meter_html()


def do_lora_voice_to_library(source, name):
    emb = load_speaker_embedding((source or "").strip()) if source else None
    if emb is None:
        return gr.update(), status_line("This adapter ships no speaker embedding.", hot=True)
    nm = (name or "lora_voice").strip()
    voices.save_voice_from_embedding(nm, emb, note=f"from {source}")
    return gr.update(choices=voices.list_voices(), value=nm), status_line(f'saved "{nm}" to library')


# ---- UI ----------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="Qwen Voice Studio", analytics_enabled=False) as demo:
        gr.HTML(theme.header_html())
        meter = gr.HTML(meter_html())

        with gr.Tabs():
            # ---- Clone ----
            with gr.Tab("Clone"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">01</span> &nbsp;clone a voice from a few seconds of audio</div>')
                with gr.Row():
                    with gr.Column():
                        c_ref = gr.Audio(label="Reference audio", type="numpy", sources=["upload", "microphone"])
                        c_reftext = gr.Textbox(label="Reference transcript", lines=2, placeholder="What the reference audio says (improves fidelity).")
                        c_xvec = gr.Checkbox(False, label="x-vector only (skip transcript, lower fidelity)")
                        c_text = gr.Textbox(label="Text to speak", lines=4, placeholder="Type what the cloned voice should say…")
                        c_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        c_long = gr.Checkbox(True, label="Long-form chunking")
                        c_adv = advanced_controls()
                        c_btn = gr.Button("Clone & Speak", variant="primary", elem_classes="qvs-generate")
                    with gr.Column():
                        c_out = gr.Audio(label="Output", type="numpy", interactive=False, autoplay=False)
                        c_status = gr.HTML(status_line("Ready."))
                c_btn.click(do_clone, [c_ref, c_reftext, c_xvec, c_text, c_lang, c_long, *c_adv], [c_out, c_status, meter])

            # ---- Preset Voices ----
            with gr.Tab("Preset Voices"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">02</span> &nbsp;nine studio voices, directed by plain language</div>')
                with gr.Row():
                    with gr.Column():
                        p_text = gr.Textbox(label="Text to speak", lines=4, placeholder="Type what to say…")
                        with gr.Row():
                            p_speaker = gr.Dropdown(SPEAKER_CHOICES, value="Ryan", label="Voice")
                            p_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        p_instruct = gr.Textbox(label="Direction (optional)", lines=2, placeholder="e.g. Very happy · Whisper softly · Angry and forceful")
                        p_examples = gr.Dropdown(["—"] + config.EMOTION_PRESETS, value="—", label="Quick directions")
                        p_long = gr.Checkbox(True, label="Long-form chunking")
                        p_adv = advanced_controls()
                        p_btn = gr.Button("Speak", variant="primary", elem_classes="qvs-generate")
                    with gr.Column():
                        p_out = gr.Audio(label="Output", type="numpy", interactive=False)
                        p_status = gr.HTML(status_line("Ready."))
                p_examples.change(lambda x: "" if x == "—" else x, p_examples, p_instruct)
                p_btn.click(do_preset, [p_text, p_speaker, p_instruct, p_lang, p_long, *p_adv], [p_out, p_status, meter])

            # ---- Voice Design ----
            with gr.Tab("Voice Design"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">03</span> &nbsp;invent a voice from a written description</div>')
                with gr.Row():
                    with gr.Column():
                        d_text = gr.Textbox(label="Text to speak", lines=4, value="It's in the top drawer… wait, it's empty? No way, that's impossible!")
                        d_instruct = gr.Textbox(label="Voice description", lines=3, placeholder="Describe the timbre, age, emotion, pace…")
                        d_examples = gr.Dropdown(["—"] + config.VOICE_DESIGN_EXAMPLES, value="—", label="Example descriptions")
                        d_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        d_long = gr.Checkbox(True, label="Long-form chunking")
                        d_adv = advanced_controls()
                        d_btn = gr.Button("Design & Speak", variant="primary", elem_classes="qvs-generate")
                    with gr.Column():
                        d_out = gr.Audio(label="Output", type="numpy", interactive=False)
                        d_status = gr.HTML(status_line("Ready."))
                d_examples.change(lambda x: "" if x == "—" else x, d_examples, d_instruct)
                d_btn.click(do_design, [d_text, d_instruct, d_lang, d_long, *d_adv], [d_out, d_status, meter])

            # ---- LoRA Lab ----
            with gr.Tab("LoRA Lab"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">04</span> &nbsp;load a fine-tuned adapter onto the base voice</div>')
                with gr.Row():
                    with gr.Column():
                        l_src = gr.Textbox(label="Adapter (HF repo id or local path)", value="loubna1101/Qwen3-TTS-Darija-LoRa",
                                           placeholder="e.g. loubna1101/Qwen3-TTS-Darija-LoRa")
                        with gr.Row():
                            l_apply = gr.Button("Apply adapter", variant="primary", elem_classes="qvs-generate")
                            l_remove = gr.Button("Remove", variant="secondary")
                        gr.HTML('<div class="qvs-eyebrow">save the adapter\'s shipped voice to your library</div>')
                        with gr.Row():
                            l_vname = gr.Textbox(label="Save voice as", value="darija_voice", scale=2)
                            l_save = gr.Button("Save voice", variant="secondary", scale=1)
                    with gr.Column():
                        l_status = gr.HTML(status_line("No adapter applied. Base is clean."))
                        gr.Markdown("After applying, use the **Clone** or **Voice Library** tab to generate — the adapter reshapes the base voice model.")
                l_apply.click(do_apply_lora, [l_src], [l_status, meter])
                l_remove.click(do_remove_lora, None, [l_status, meter])

            # ---- Voice Library ----
            with gr.Tab("Voice Library"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">05</span> &nbsp;save voices once, reuse them everywhere</div>')
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="qvs-eyebrow">save a new voice from reference audio</div>')
                        v_name = gr.Textbox(label="Voice name", placeholder="e.g. narrator")
                        v_ref = gr.Audio(label="Reference audio", type="numpy", sources=["upload", "microphone"])
                        v_reftext = gr.Textbox(label="Reference transcript", lines=2)
                        v_xvec = gr.Checkbox(False, label="x-vector only")
                        v_save = gr.Button("Save to library", variant="secondary")
                    with gr.Column():
                        gr.HTML('<div class="qvs-eyebrow">speak with a saved voice</div>')
                        v_pick = gr.Dropdown(voices.list_voices(), label="Saved voices")
                        v_refresh = gr.Button("↻ refresh", variant="secondary", scale=0)
                        v_text = gr.Textbox(label="Text to speak", lines=3)
                        v_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        v_long = gr.Checkbox(True, label="Long-form chunking")
                        v_adv = advanced_controls()
                        v_btn = gr.Button("Speak", variant="primary", elem_classes="qvs-generate")
                        v_out = gr.Audio(label="Output", type="numpy", interactive=False)
                        v_status = gr.HTML(status_line("Ready."))
                v_save.click(do_save_voice, [v_name, v_ref, v_reftext, v_xvec], [v_pick, v_status])
                v_refresh.click(lambda: gr.update(choices=voices.list_voices()), None, v_pick)
                v_btn.click(do_library_gen, [v_pick, v_text, v_lang, v_long, *v_adv], [v_out, v_status, meter])
                # LoRA -> library bridge lives here too
                l_save.click(do_lora_voice_to_library, [l_src, l_vname], [v_pick, l_status])

        gr.HTML(theme.footer_html())
        timer = gr.Timer(4.0)
        timer.tick(meter_html, None, meter)
    return demo


if __name__ == "__main__":
    demo = build()
    demo.queue(default_concurrency_limit=int(os.environ.get("QVS_CONCURRENCY", "4")))
    demo.launch(
        theme=theme.studio_theme(),
        css=theme.CSS,
        server_name=os.environ.get("QVS_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("QVS_PORT", "7860")),
        show_error=True,
    )
