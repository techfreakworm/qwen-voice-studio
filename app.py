"""Qwen Voice Studio — Gradio app (runs on local MPS and Hugging Face ZeroGPU).

Five channels over the three Qwen3-TTS-12Hz-1.7B checkpoints: Clone, Preset
Voices, Voice Design, LoRA Lab (management only), and a Voice Library that ties
them together. One codebase, two platforms; see docs/DESIGN.md.
"""
from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np

from qvs import audio as qaudio
from qvs import config, engine, voices
from qvs.device import get_attn_impl, gpu, on_zerogpu, target_device
from qvs.lora import AdapterManager, load_speaker_embedding
from qvs.memory import MemoryGuard, snapshot
from qvs.registry import ModelRegistry
from qvs.ui import theme

REG = ModelRegistry()
# LoRA is applied per-generation *inside* the @spaces.GPU fork: ZeroGPU forks do
# not persist in-place model mutations across requests, so "apply once, use later"
# can't work there. We track only the selected adapter here (a plain string).
SELECTED_ADAPTER = {"source": ""}


def _apply_adapter(model, source: str):
    """Attach the adapter to a fresh manager (caller must .unload after gen)."""
    if not (source or "").strip():
        return None
    mgr = AdapterManager()
    mgr.apply(model, source.strip())
    return mgr

# NOTE: on ZeroGPU we deliberately do NOT preload at module level. A 14 GB
# download + all-3 load + tensor-packing at import overran the Space startup
# window (RUNTIME_ERROR before the server ever answered a health check). Instead
# the Gradio server starts instantly and each checkpoint loads lazily on the
# first @spaces.GPU request for its mode (one at a time — bounds container RAM).

NONE_VOICE = "— none —"
LANG_CHOICES = list(config.LANGUAGES.keys())
SPEAKER_CHOICES = [(f"{s.display} — {s.description.rstrip('.')} ({s.language})", s.key) for s in config.SPEAKERS]

# RAM watchdog protects the local macOS gate (DESIGN §6). On ZeroGPU the
# constraint is the 48 GB card, not container RAM (where psutil misreports),
# so the RAM guard is disabled there.
if not on_zerogpu():
    MemoryGuard(hard_gb=float(os.environ.get("QVS_MEMGUARD_HARD", "76")),
                soft_gb=float(os.environ.get("QVS_MEMGUARD_SOFT", "72"))).start()


# ---- helpers -----------------------------------------------------------------
def meter_html() -> str:
    if SELECTED_ADAPTER["source"]:
        lora = f' · LoRA <b>{SELECTED_ADAPTER["source"].split("/")[-1]}</b>'
    else:
        lora = ""
    if on_zerogpu():
        mem = "MEM <b>ZeroGPU</b>"  # container RAM is not the constraint here
    else:
        snap = snapshot()
        mem = f"MEM <b>{snap.committed:.0f}</b>/{snap.total:.0f} GB"
    return (
        f'<div class="qvs-meter">DEVICE <b>{target_device()}</b> · DTYPE <b>bf16</b> · '
        f'ATTN <b>{get_attn_impl()}</b> · {mem} · '
        f'RESIDENT <b>{len(REG.loaded)}</b>/3{lora}</div>'
    )


def gp(a) -> engine.GenParams:
    return engine.GenParams(
        temperature=float(a[0]), top_p=float(a[1]), top_k=int(a[2]), repetition_penalty=float(a[3]),
        subtalker_dosample=bool(a[4]),
        subtalker_temperature=float(a[5]), subtalker_top_p=float(a[6]), subtalker_top_k=int(a[7]),
        max_new_tokens=int(a[8]), seed=int(a[9]),
    )


def advanced_controls():
    d = config.GEN_DEFAULTS
    with gr.Accordion("Advanced — sampling & sub-talker", open=False):
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, d.temperature, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, d.top_p, step=0.05, label="Top-p")
            top_k = gr.Slider(0, 100, d.top_k, step=1, label="Top-k")
            repetition_penalty = gr.Slider(1.0, 2.0, d.repetition_penalty, step=0.01, label="Repetition penalty")
        with gr.Row():
            st_dosample = gr.Checkbox(True, label="Sub-talker sampling")
            st_temp = gr.Slider(0.0, 1.5, d.subtalker_temperature, step=0.05, label="Sub-talker temp")
            st_top_p = gr.Slider(0.0, 1.0, d.subtalker_top_p, step=0.05, label="Sub-talker top-p")
            st_top_k = gr.Slider(0, 100, d.subtalker_top_k, step=1, label="Sub-talker top-k")
        with gr.Row():
            max_new = gr.Slider(128, 4096, d.max_new_tokens, step=64, label="Max new tokens")
            seed = gr.Number(d.seed, precision=0, label="Seed (-1 = random)")
    return [temperature, top_p, top_k, repetition_penalty, st_dosample, st_temp, st_top_p, st_top_k, max_new, seed]


def status_line(msg: str, hot: bool = False) -> str:
    return f'<div class="qvs-status {"on" if hot else ""}">{msg}</div>'


def _done(t0: float, wav) -> str:
    return status_line(f"done · {len(wav)/config.OUTPUT_SAMPLE_RATE:.1f}s audio in {time.time()-t0:.1f}s")


def _adapter_report(info) -> str:
    warn = ' · <span style="color:#FF6B4A">⚠ base mismatch</span>' if info.base_mismatch else ""
    emb = " · ships a voice" if info.has_speaker_embedding else ""
    return status_line(
        f"attached <b>{info.source.split('/')[-1]}</b> · r={info.r} α={info.alpha} · "
        f"{info.n_modules} modules on {', '.join(t.replace('_proj','') for t in (info.target_modules or []))}{emb}{warn}"
    )


# ---- callbacks (decorated for ZeroGPU; no-op locally) ------------------------
@gpu(duration=120)
def do_preset(text, speaker, instruct, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter some text to speak.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("custom_voice")
    wav, sr = engine.synth_custom_voice(model, text.strip(), speaker, instruct, config.LANGUAGES[language], gp(adv), bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


@gpu(duration=120)
def do_design(text, instruct, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter some text to speak.", hot=True), meter_html()
    if not (instruct or "").strip():
        return None, status_line("Describe the voice you want to design.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("voice_design")
    wav, sr = engine.synth_voice_design(model, text.strip(), instruct.strip(), config.LANGUAGES[language], gp(adv), bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


@gpu(duration=120)
def do_clone(ref_audio, ref_text, xvec, voice_pick, adapter_source, text, language, longform, *adv):
    if not (text or "").strip():
        return None, status_line("Enter text to synthesize.", hot=True), meter_html()
    t0 = time.time()
    model = REG.to_device("base")
    lora = None
    try:
        if (adapter_source or "").strip():
            try:
                lora = _apply_adapter(model, adapter_source)
            except Exception as e:
                return None, status_line(f"adapter error: {type(e).__name__}: {e}", hot=True), meter_html()
        if voice_pick and voice_pick != NONE_VOICE:
            items = voices.load_voice(voice_pick)
            wav, sr = engine.synth_clone(model, text.strip(), config.LANGUAGES[language], gp(adv),
                                         voice_clone_prompt=items, longform=bool(longform))
        else:
            ref = qaudio.ref_from_gradio(ref_audio)
            if ref is None:
                return None, status_line("Upload reference audio or pick a saved voice.", hot=True), meter_html()
            if not xvec and not (ref_text or "").strip():
                return None, status_line("Add the reference transcript, or enable x-vector-only.", hot=True), meter_html()
            wav, sr = engine.synth_clone(model, text.strip(), config.LANGUAGES[language], gp(adv),
                                         ref_audio=ref, ref_text=(ref_text or None), x_vector_only=bool(xvec), longform=bool(longform))
    finally:
        if lora is not None:
            try:
                lora.unload(model)
            except Exception:
                pass
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


@gpu(duration=120)
def do_library_gen(voice_name, text, language, longform, *adv):
    if not voice_name or voice_name == NONE_VOICE:
        return None, status_line("Pick a saved voice.", hot=True), meter_html()
    if not (text or "").strip():
        return None, status_line("Enter text to speak.", hot=True), meter_html()
    t0 = time.time()
    items = voices.load_voice(voice_name)
    model = REG.to_device("base")
    wav, sr = engine.synth_clone(model, text.strip(), config.LANGUAGES[language], gp(adv),
                                 voice_clone_prompt=items, longform=bool(longform))
    return qaudio.to_gradio(wav, sr), _done(t0, wav), meter_html()


@gpu(duration=120)
def do_lora_quicktest(source, sentence):
    src = (source or "").strip()
    if not src:
        return None, status_line("Enter an adapter (repo id or path) above first.", hot=True)
    emb = load_speaker_embedding(src)
    if emb is None:
        return None, status_line("This adapter ships no voice — use it in the Clone tab with your own reference.", hot=True)
    import torch
    from qwen_tts import VoiceClonePromptItem
    model = REG.to_device("base")
    lora = None
    try:
        lora = _apply_adapter(model, src)
        item = VoiceClonePromptItem(ref_code=None,
                                    ref_spk_embedding=torch.as_tensor(emb).to(model.device).to(torch.bfloat16),
                                    x_vector_only_mode=True, icl_mode=False, ref_text=None)
        wav, sr = engine.synth_clone(model, sentence.strip() or "Hello from the adapter.", "Auto",
                                     engine.GenParams(max_new_tokens=512), voice_clone_prompt=[item], longform=False)
        return qaudio.to_gradio(wav, sr), status_line("quick test done")
    except Exception as e:
        return None, status_line(f"quick test failed: {type(e).__name__}: {e}", hot=True)
    finally:
        if lora is not None:
            try:
                lora.unload(model)
            except Exception:
                pass


# management callbacks — validate/inspect only (adapter is applied per generation)
def do_apply_lora(source):
    src = (source or "").strip()
    if not src:
        return status_line("Enter a Hugging Face repo id or local path.", hot=True), meter_html()
    try:
        from qvs.lora import read_adapter_config, resolve_adapter
        cfg = read_adapter_config(resolve_adapter(src))
    except Exception as e:
        return status_line(f"Couldn't load adapter: {type(e).__name__}: {e}", hot=True), meter_html()
    SELECTED_ADAPTER["source"] = src
    has_emb = load_speaker_embedding(src) is not None
    targets = ", ".join(t.replace("_proj", "") for t in (cfg.get("target_modules") or []))
    emb = " · ships a voice" if has_emb else ""
    return (status_line(f"selected <b>{src.split('/')[-1]}</b> · r={cfg.get('r')} α={cfg.get('lora_alpha')} · "
                        f"{targets}{emb} — applied per generation (Clone tab or Quick test)"), meter_html())


def do_unload_lora():
    SELECTED_ADAPTER["source"] = ""
    return status_line("adapter cleared."), meter_html()


@gpu(duration=90)
def do_save_voice(name, ref_audio, ref_text, xvec):
    if not (name or "").strip():
        return status_line("Give the voice a name.", hot=True)
    ref = qaudio.ref_from_gradio(ref_audio)
    if ref is None:
        return status_line("Upload reference audio to save.", hot=True)
    if not xvec and not (ref_text or "").strip():
        return status_line("Reference transcript required (or enable x-vector-only).", hot=True)
    voices.save_voice(REG.to_device("base"), name.strip(), ref, (ref_text or None), bool(xvec))
    return status_line(f'saved voice "{name.strip()}"')


def do_lora_voice_to_library(source, name):
    emb = load_speaker_embedding((source or "").strip()) if source else None
    if emb is None:
        return status_line("This adapter ships no speaker embedding.", hot=True)
    voices.save_voice_from_embedding((name or "lora_voice").strip(), emb, note=f"from {source}")
    return status_line(f'saved "{(name or "lora_voice").strip()}" to library')


@gpu(duration=90)
def do_design_to_library(design_audio, design_text, name):
    if design_audio is None:
        return status_line("Generate a designed voice first.", hot=True)
    if not (name or "").strip():
        return status_line("Name the voice to save it.", hot=True)
    sr, data = design_audio
    ref = (np.asarray(data, dtype=np.float32), int(sr))
    voices.save_voice(REG.to_device("base"), name.strip(), ref, (design_text or None), x_vector_only=False,
                      note="from Voice Design")
    return status_line(f'saved designed voice "{name.strip()}" — use it in Clone or Voice Library')


# ---- UI ----------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="Qwen Voice Studio", analytics_enabled=False) as demo:
        gr.HTML(theme.header_html())
        meter = gr.HTML(meter_html())
        voice_pickers: list = []  # refreshed together on save

        with gr.Tabs():
            # ---- Clone ----
            with gr.Tab("Clone"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">01</span> &nbsp;clone a voice from a few seconds of audio</div>')
                with gr.Row():
                    with gr.Column():
                        c_ref = gr.Audio(label="Reference audio", type="numpy", sources=["upload", "microphone"])
                        c_reftext = gr.Textbox(label="Reference transcript", lines=2, placeholder="What the reference says (improves fidelity).")
                        c_xvec = gr.Checkbox(False, label="x-vector only (skip transcript, lower fidelity)")
                        c_voice = gr.Dropdown([NONE_VOICE] + voices.list_voices(), value=NONE_VOICE, label="…or use a saved voice")
                        c_adapter = gr.Textbox(label="LoRA adapter (optional — HF repo id)", placeholder="e.g. loubna1101/Qwen3-TTS-Darija-LoRa")
                        c_text = gr.Textbox(label="Text to speak", lines=4, placeholder="Type what the cloned voice should say…")
                        c_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        c_long = gr.Checkbox(True, label="Long-form chunking")
                        c_adv = advanced_controls()
                        c_btn = gr.Button("Clone & Speak", variant="primary", elem_classes="qvs-generate")
                    with gr.Column():
                        c_out = gr.Audio(label="Output", type="numpy", interactive=False)
                        c_status = gr.HTML(status_line("Ready."))
                voice_pickers.append(c_voice)
                c_btn.click(do_clone, [c_ref, c_reftext, c_xvec, c_voice, c_adapter, c_text, c_lang, c_long, *c_adv], [c_out, c_status, meter])

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
                        gr.HTML('<div class="qvs-eyebrow">Design → Clone bridge — lock this voice in for reuse</div>')
                        with gr.Row():
                            d_savename = gr.Textbox(label="Save designed voice as", scale=2, placeholder="e.g. narrator")
                            d_save = gr.Button("Send to Library", variant="secondary", scale=1)
                d_examples.change(lambda x: "" if x == "—" else x, d_examples, d_instruct)
                d_btn.click(do_design, [d_text, d_instruct, d_lang, d_long, *d_adv], [d_out, d_status, meter])

            # ---- LoRA Lab (management only) ----
            with gr.Tab("LoRA Lab"):
                gr.HTML('<div class="qvs-eyebrow"><span class="num">04</span> &nbsp;load a fine-tuned adapter onto the Base voice</div>')
                with gr.Row():
                    with gr.Column():
                        l_src = gr.Textbox(label="Adapter (HF repo id or local path)", value="loubna1101/Qwen3-TTS-Darija-LoRa")
                        with gr.Row():
                            l_apply = gr.Button("Load & inspect", variant="primary", elem_classes="qvs-generate")
                            l_remove = gr.Button("Clear", variant="secondary")
                        gr.HTML('<div class="qvs-eyebrow">save the adapter\'s bundled voice to your library</div>')
                        with gr.Row():
                            l_vname = gr.Textbox(label="Save voice as", value="darija_voice", scale=2)
                            l_save = gr.Button("Save voice", variant="secondary", scale=1)
                    with gr.Column():
                        l_status = gr.HTML(status_line("No adapter applied. Base is clean."))
                        gr.HTML('<div class="qvs-eyebrow">quick test (uses the adapter\'s bundled voice)</div>')
                        l_testtext = gr.Textbox(label="Test sentence", value="Salam, hada ikhtibar dyal les voix.", lines=2)
                        l_testbtn = gr.Button("Quick test", variant="secondary")
                        l_testout = gr.Audio(label="Quick test output", type="numpy", interactive=False)
                l_apply.click(do_apply_lora, [l_src], [l_status, meter])
                l_remove.click(do_unload_lora, None, [l_status, meter])
                l_testbtn.click(do_lora_quicktest, [l_src, l_testtext], [l_testout, l_status])

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
                        with gr.Row():
                            v_pick = gr.Dropdown([NONE_VOICE] + voices.list_voices(), value=NONE_VOICE, label="Saved voices", scale=3)
                            v_refresh = gr.Button("↻", variant="secondary", scale=1)
                        v_text = gr.Textbox(label="Text to speak", lines=3)
                        v_lang = gr.Dropdown(LANG_CHOICES, value="Auto (detect)", label="Language")
                        v_long = gr.Checkbox(True, label="Long-form chunking")
                        v_adv = advanced_controls()
                        v_btn = gr.Button("Speak", variant="primary", elem_classes="qvs-generate")
                        v_out = gr.Audio(label="Output", type="numpy", interactive=False)
                        v_status = gr.HTML(status_line("Ready."))
                voice_pickers.append(v_pick)

                # wire saves to refresh every voice picker (Clone + Library)
                v_save.click(do_save_voice, [v_name, v_ref, v_reftext, v_xvec], [v_status]).then(
                    lambda: [gr.update(choices=[NONE_VOICE] + voices.list_voices()) for _ in voice_pickers], None, voice_pickers)
                d_save.click(do_design_to_library, [d_out, d_text, d_savename], [d_status]).then(
                    lambda: [gr.update(choices=[NONE_VOICE] + voices.list_voices()) for _ in voice_pickers], None, voice_pickers)
                l_save.click(do_lora_voice_to_library, [l_src, l_vname], [l_status]).then(
                    lambda: [gr.update(choices=[NONE_VOICE] + voices.list_voices()) for _ in voice_pickers], None, voice_pickers)
                v_refresh.click(lambda: gr.update(choices=[NONE_VOICE] + voices.list_voices()), None, v_pick)
                v_btn.click(do_library_gen, [v_pick, v_text, v_lang, v_long, *v_adv], [v_out, v_status, meter])

        gr.HTML(theme.footer_html())
        gr.Timer(4.0).tick(meter_html, None, meter)
    return demo


if __name__ == "__main__":
    demo = build()
    demo.queue(default_concurrency_limit=1)  # one model, one device — serialize (DESIGN §6)
    launch_kwargs = dict(theme=theme.studio_theme(), css=theme.CSS, show_error=True, ssr_mode=False)
    if on_zerogpu():
        launch_kwargs["server_name"] = "0.0.0.0"  # HF health check must reach the app
    else:
        launch_kwargs["server_name"] = os.environ.get("QVS_HOST", "127.0.0.1")
        launch_kwargs["server_port"] = int(os.environ.get("QVS_PORT", "7860"))
    demo.launch(**launch_kwargs)
