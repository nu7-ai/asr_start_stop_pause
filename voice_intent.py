import argparse
import json
import queue
import re
import sys
import threading
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer


# -------- Intent Resolver (rule-based, lightweight) --------
_START_PATTERNS = (
    r"\bstart\b", r"\bbegin\b", r"\bcommence\b", r"\binitiate\b",
    r"\bresume\b", r"\bcontinue\b", r"\bplay\b", r"\bgo ahead\b", r"\bkick off\b",
    r"\bstart the (?:session|process)\b", r"\bplease start\b"
)

_STOP_PATTERNS = (
    r"\bstop\b", r"\bend\b", r"\bhalt\b", r"\babort\b", r"\bterminate\b",
    r"\bquit\b", r"\bcancel\b", r"\bshut\s*(?:down|off)\b",
    r"\b(end|stop) the (?:session|process)\b", r"\bplease stop\b", r"\bstop it\b"
)

_PAUSE_PATTERNS = (
    r"\bpause\b", r"\bhold\b", r"\bhold on\b", r"\bhang on\b",
    r"\bwait\b", r"\bfreeze\b", r"\bsuspend\b",
    r"\bpause the (?:session|process)\b", r"\bplease pause\b"
)

_EXIT_PATTERNS = (
    r"\bexit\b", r"\bquit (?:asr|listening|program)\b", r"\bclose\b",
    r"\bstop listening\b", r"\bexit (?:the )?(?:asr|program|app)\b",
    r"\bend listening\b", r"\bterminate (?:asr|listening)\b", r"\bgoodbye\b"
)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def match_any(text: str, patterns) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def resolve_intent(text: str) -> str:
    """
    Map recognized text to: EXIT / STOP / PAUSE / START / OTHER
    Priority: EXIT > STOP > PAUSE > START
    """
    t = normalize(text)
    if match_any(t, _EXIT_PATTERNS):
        return "EXIT"
    if match_any(t, _STOP_PATTERNS):
        return "STOP"
    if match_any(t, _PAUSE_PATTERNS):
        return "PAUSE"
    if match_any(t, _START_PATTERNS):
        return "START"
    return "OTHER"


# -------- Audio + ASR --------
def audio_stream(q: queue.Queue, samplerate: int, channels: int, device: int | None, blocksize: int = 4096):
    def callback(indata, frames, time_info, status):
        if status:
            # You could print(status) for debugging
            pass
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        dtype="int16",
        channels=channels,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)  # callback does the work


def pick_samplerate(device: int | None, fallback: int | None) -> int:
    if fallback:
        return int(fallback)
    try:
        dev = sd.query_devices(device, "input")
        # Prefer the device's default sample rate (often 44100 or 48000)
        sr = int(dev["default_samplerate"])
        return sr if sr > 0 else 16000
    except Exception:
        return 16000


def main():
    parser = argparse.ArgumentParser(description="Ultra-light voice intent recognizer (START/STOP/PAUSE/EXIT/OTHER)")
    parser.add_argument("--model", required=True, help="Path to Vosk model folder (unzipped).")
    parser.add_argument("--samplerate", type=int, default=48000, help="Sampling rate (defaults to mic's native).")
    parser.add_argument("--device", type=int, default=None, help="Sounddevice input device index (optional).")
    parser.add_argument("--silence-timeout", type=float, default=2.0,
                        help="Seconds to auto-exit after no speech (0 = never).")
    parser.add_argument("--debug", action="store_true", help="Print raw partial/final transcripts.")
    args = parser.parse_args()

    samplerate = pick_samplerate(args.device, args.samplerate)
    model = Model(args.model)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(False)

    q = queue.Queue()
    audio_thread = threading.Thread(target=audio_stream,
                                    args=(q, samplerate, 1, args.device),
                                    daemon=True)
    audio_thread.start()

    print(f"Listening continuously at {samplerate} Hz… Say: start / stop / pause. Say 'exit' to stop ASR. Ctrl+C also works.\n")

    last_intent = None
    last_activity_ts = time.time()
    last_emit_ts = 0.0
    debounce_sec = 0.6  # suppress duplicate prints from partials

    try:
        while True:
            try:
                data = q.get(timeout=0.8)
            except queue.Empty:
                if args.silence_timeout > 0 and (time.time() - last_activity_ts) > args.silence_timeout:
                    print("No speech detected for set timeout. Exiting.")
                    break
                continue

            got_final = rec.AcceptWaveform(data)
            now = time.time()

            if got_final:
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if not text:
                    continue

                last_activity_ts = now
                if args.debug:
                    print(f"(final) {text}")

                intent = resolve_intent(text)
                if intent != "OTHER":
                    print(f"[{time.strftime('%H:%M:%S')}] HEARD: \"{text}\" -> INTENT: {intent}")
                    last_intent = intent
                    last_emit_ts = now
                    if intent == "EXIT":
                        print("Exit requested by voice. Stopping ASR.")
                        break
                else:
                    # Print 'OTHER' only when it follows a different intent to avoid spam
                    if last_intent != "OTHER":
                        print(f"[{time.strftime('%H:%M:%S')}] HEARD: \"{text}\" -> INTENT: OTHER")
                        last_intent = "OTHER"

            else:
                # Low-latency: check partials too
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if not partial:
                    continue

                last_activity_ts = now
                if args.debug:
                    print(f"(partial) {partial}")

                intent = resolve_intent(partial)
                if intent != "OTHER":
                    # Debounce duplicates from rolling partials
                    if (now - last_emit_ts) > debounce_sec or intent != last_intent:
                        print(f"[{time.strftime('%H:%M:%S')}] HEARD: \"{partial}\" -> INTENT: {intent}")
                        last_intent = intent
                        last_emit_ts = now

                        # If we acted on a command, you can Reset() to avoid echoing it again
                        if intent == "EXIT":
                            print("Exit requested by voice. Stopping ASR.")
                            break
                        # rec.Reset()  # Uncomment if you want to flush partial state after an action

    except KeyboardInterrupt:
        print("\nExiting (keyboard interrupt).")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()