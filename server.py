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
        return word, p, False  # punctuation / whitespace: already a whole word
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
            return self._json({"models": local, "default": self.server.model})
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

        try:
            stream = ollama({
                "model": model, "prompt": prompt, "raw": True, "stream": True,
                "logprobs": True, "top_logprobs": TOP_K, "options": options,
            })
            i = 0
            for line in stream:
                if aborted.is_set():
                    break
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for lp in chunk.get("logprobs") or []:
                    tok = {
                        "token": token_text(lp),
                        "p": math.exp(lp["logprob"]),
                        "alts": [
                            {"token": token_text(t), "p": math.exp(t["logprob"])}
                            for t in lp.get("top_logprobs", [])
                        ],
                    }
                    cur_word = "".join(t["token"] for t in word_toks)
                    if not word_toks or starts_new_word(cur_word, tok["token"]):
                        close_word()
                        prev_word = cur_word
                        w_idx += 1
                        word_toks = []
                        word_start_ctx = text
                    word_toks.append(tok)
                    text += tok["token"]
                    send({"type": "token", "i": i, "w": w_idx, **tok})
                    i += 1
                if chunk.get("done"):
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
