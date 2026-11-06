#!/usr/bin/env python3
"""Next-token prediction playground server.

Serves index.html and proxies completion requests to a local Ollama
instance, streaming back each generated token with its top alternatives,
plus word-level probabilities and alternative words.

Usage: python3 server.py [--port 8000] [--model qwen2.5:1.5b]
"""
import argparse
import json
import math
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OLLAMA = "http://localhost:11434"
ROOT = Path(__file__).parent
TOP_K = 5
MAX_WORD_TOKENS = 5  # cap when greedily completing an alternative into a word
POOL = ThreadPoolExecutor(max_workers=4)


def token_text(entry):
    """Decode token from its bytes when present (handles multi-byte UTF-8)."""
    b = entry.get("bytes")
    if b:
        return bytes(b).decode("utf-8", errors="replace")
    return entry.get("token", "")


def starts_new_word(prev_word, tok):
    """Does `tok` begin a new word, given the text of the word so far?"""
    if not prev_word or not tok:
        return True
    c = tok[0]
    if c.isspace():
        return True
    has_alnum = any(ch.isalnum() for ch in prev_word)
    if c.isalnum():
        return not has_alnum  # letters after punctuation/whitespace start a word
    if c in "'’-" and has_alnum and prev_word[-1].isalnum():
        return False  # don't, well-known
    return True  # punctuation is its own word


END_MARKERS = {"", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "<eos>", "<|end_of_text|>"}


def is_end(tok_text):
    """End-of-turn / end-of-text tokens come back from Ollama as empty text."""
    return tok_text in END_MARKERS


# Chat templates, written out explicitly so the UI can show exactly what the
# model receives. Detected per model from Ollama's template / model family.
def render_chat(fmt, system, user):
    if fmt == "chatml":
        sys_part = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return f"{sys_part}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if fmt == "llama3":
        sys_part = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>" if system else ""
        return (f"{sys_part}<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    if fmt == "gemma":
        sys_part = f"{system}\n\n" if system else ""
        return f"<start_of_turn>user\n{sys_part}{user}<end_of_turn>\n<start_of_turn>model\n"
    raise ValueError(f"no chat format: {fmt}")


_FORMATS = {}


def chat_format(model):
    if model not in _FORMATS:
        req = urllib.request.Request(f"{OLLAMA}/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        info = json.load(urllib.request.urlopen(req))
        tpl, fam = info.get("template", ""), (info.get("details") or {}).get("family", "")
        has_tpl = ".Messages" in tpl or "<|im_start|>" in tpl or "<start_of_turn>" in tpl or "<|start_header_id|>" in tpl
        if "<|im_start|>" in tpl or fam.startswith("qwen"):
            fmt = "chatml"
        elif "<|start_header_id|>" in tpl or fam == "llama":
            fmt = "llama3"
        elif "<start_of_turn>" in tpl or fam.startswith("gemma"):
            fmt = "gemma"
        else:
            fmt = None
        # A model whose Ollama template is just {{ .Prompt }} was not set up for chat (base model)
        _FORMATS[model] = {"format": fmt, "instruct": has_tpl}
    return _FORMATS[model]


def probe_end(model, text):
    """Ollama stops on the end-of-turn token without reporting its probability.
    Re-run that last step with a huge temperature so a *different* token is
    almost surely sampled; its top_logprobs (which are pre-temperature) then
    include the end token with its true probability."""
    for seed in range(4):
        d = json.load(ollama({
            "model": model, "prompt": text, "raw": True, "stream": False,
            "logprobs": True, "top_logprobs": TOP_K,
            "options": {"num_predict": 1, "temperature": 100, "top_k": 40, "top_p": 1, "seed": seed},
        }))
        lps = d.get("logprobs") or []
        if lps:
            alts = [{"token": token_text(t), "p": math.exp(t["logprob"])} for t in lps[0].get("top_logprobs", [])]
            for a in alts:
                if is_end(a["token"]):
                    a["token"] = ""
            end = next((a for a in alts if a["token"] == ""), None)
            return (end["p"] if end else None), alts
    return None, []


def ollama(req):
    r = urllib.request.urlopen(
        urllib.request.Request(
            f"{OLLAMA}/api/generate",
            data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"},
        )
    )
    return r


def complete_word(model, context, first_tok, first_p):
    """Greedily extend a candidate first token until the word ends."""
    word, p = first_tok, first_p
    if not any(ch.isalnum() for ch in first_tok):
        return word, p, False  # punctuation / whitespace / end token: already whole
    d = json.load(ollama({
        "model": model, "prompt": context + first_tok, "raw": True, "stream": False,
        "logprobs": True, "options": {"num_predict": MAX_WORD_TOKENS, "temperature": 0},
    }))
    lps = d.get("logprobs") or []
    for lp in lps:
        t = token_text(lp)
        if starts_new_word(word, t):
            return word, p, False
        word += t
        p *= math.exp(lp["logprob"])
    return word, p, len(lps) >= MAX_WORD_TOKENS  # truncated


def word_alternatives(model, context, word_tokens, prev_word):
    """Word-level candidates for a finished word: the chosen word plus each
    top first-token alternative completed into a whole word.

    Alternatives that would *continue* the previous word (e.g. "ed" after
    "orphan") are not candidates for this word, so they are skipped."""
    first = word_tokens[0]
    chosen_text = "".join(t["token"] for t in word_tokens)
    chosen_p = math.prod(t["p"] for t in word_tokens)
    cands = {chosen_text: {"word": chosen_text, "p": chosen_p, "chosen": True}}
    for alt in first["alts"]:
        if alt["token"] == first["token"]:
            continue  # this branch is the chosen word itself
        if prev_word and not starts_new_word(prev_word, alt["token"]):
            continue  # would extend the previous word, not start a new one
        w, p, trunc = complete_word(model, context, alt["token"], alt["p"])
        if w in cands:
            cands[w]["p"] = max(cands[w]["p"], p)
        else:
            cands[w] = {"word": w, "p": p, "chosen": False, "truncated": trunc}
    return chosen_p, sorted(cands.values(), key=lambda c: -c["p"])


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/api/models":
            tags = json.load(urllib.request.urlopen(f"{OLLAMA}/api/tags"))
            local = sorted(m["name"] for m in tags["models"] if not m.get("remote_host"))
            return self._json({"models": [{"name": m, **chat_format(m)} for m in local],
                               "default": self.server.model})
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/complete":
            return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body.get("model") or self.server.model
        if body.get("mode") == "chat":
            fmt = chat_format(model)["format"]
            if not fmt:
                return self._json({"error": f"{model} has no known chat template"}, 400)
            prompt = render_chat(fmt, (body.get("system") or "").strip(), body["prompt"])
        else:
            prompt = body["prompt"]
        options = {
            "num_predict": int(body.get("max_tokens", 60)),
            "temperature": float(body.get("temperature", 0.8)),
        }
        if body.get("seed") is not None:
            options["seed"] = int(body["seed"])

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        lock = threading.Lock()
        aborted = threading.Event()

        def send(obj):
            if aborted.is_set():
                return
            try:
                with lock:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                aborted.set()

        def expand(w_idx, context, toks, prev_word):
            if aborted.is_set():
                return
            try:
                p, alts = word_alternatives(model, context, toks, prev_word)
                send({"type": "word", "w": w_idx, "p": p, "alts": alts})
            except Exception as e:
                send({"type": "word", "w": w_idx, "error": str(e)})

        futures = []
        text = prompt          # everything generated so far (context for alternatives)
        word_start_ctx = None  # context right before the current word
        word_toks = []
        w_idx = -1
        prev_word = ""  # text of the word before the current one

        def close_word():
            if word_toks:
                futures.append(POOL.submit(expand, w_idx, word_start_ctx, list(word_toks), prev_word))

        def make_tok(tok_text, p, alts):
            for a in alts:
                if is_end(a["token"]):
                    a["token"] = ""
            return {"token": "" if is_end(tok_text) else tok_text, "p": p, "alts": alts,
                    "end": is_end(tok_text)}

        def add_token(tok):
            nonlocal w_idx, word_toks, word_start_ctx, prev_word, text, i
            cur_word = "".join(t["token"] for t in word_toks)
            if not word_toks or tok["end"] or word_toks[-1]["end"] or starts_new_word(cur_word, tok["token"]):
                close_word()
                prev_word = cur_word
                w_idx += 1
                word_toks = []
                word_start_ctx = text
            word_toks.append(tok)
            text += tok["token"]
            send({"type": "token", "i": i, "w": w_idx, **tok})
            i += 1

        i = 0
        send({"type": "prompt", "text": prompt})
        try:
            stream = ollama({
                "model": model, "prompt": prompt, "raw": True, "stream": True,
                "logprobs": True, "top_logprobs": TOP_K, "options": options,
            })
            for line in stream:
                if aborted.is_set():
                    break
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for lp in chunk.get("logprobs") or []:
                    add_token(make_tok(
                        token_text(lp), math.exp(lp["logprob"]),
                        [{"token": token_text(t), "p": math.exp(t["logprob"])}
                         for t in lp.get("top_logprobs", [])],
                    ))
                if chunk.get("done"):
                    if chunk.get("done_reason") == "stop" and not (word_toks and word_toks[-1]["end"]):
                        # The model chose to end: recover that token's probability.
                        p_end, alts = probe_end(model, text)
                        if p_end is not None:
                            add_token(make_tok("", p_end, alts))
                    close_word()
                    send({"type": "gen_done", "reason": chunk.get("done_reason")})
                    break
            for f in futures:
                f.result()
            send({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            aborted.set()
        except Exception as e:  # surface Ollama errors to the UI
            send({"type": "error", "error": str(e)})
        finally:
            aborted.set()  # stop any queued expansions for this request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.model = args.model
    print(f"Playground: http://127.0.0.1:{args.port}  (default model: {args.model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
