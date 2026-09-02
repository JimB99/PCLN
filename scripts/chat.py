"""Interactive text-based chat with PCLN.

Load a trained model and chat in natural language.
Supports session episodic memory, repetition penalty, and optional online learning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_pcln
from src.model.generation import (
    apply_repetition_penalty,
    ban_repeating_ngrams,
    sample_next_token,
)
from src.data.text_codec import CharCodec, WordCodec

DEFAULT_CHECKPOINT = ROOT / "results" / "sprint" / "wiki_stride64" / "best_model.pt"
FALLBACK_CHECKPOINT = ROOT / "results" / "sprint" / "wiki_baseline" / "best_model.pt"


def _default_checkpoint_path() -> Path:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    if FALLBACK_CHECKPOINT.exists():
        return FALLBACK_CHECKPOINT
    return ROOT / "checkpoints" / "best_model.pt"


def _chat_unavailable_message(checkpoint: Path) -> str:
    lines = [
        "This checkpoint cannot chat in natural language.",
        f"  Path: {checkpoint}",
        "",
        "Common causes:",
        "  - Trained with --dataset dummy (Quick Start smoke test only)",
        "  - Missing vocab_mappings in the checkpoint file",
        "",
        "Use a text-trained model, for example:",
        f"  python scripts/chat.py --checkpoint {DEFAULT_CHECKPOINT}",
        f"  python scripts/chat.py --checkpoint {FALLBACK_CHECKPOINT}",
        f"  python scripts/chat.py --checkpoint results/sprint/shakespeare_char/best_model.pt",
    ]
    return "\n".join(lines)


class ChatBot:
    """Text-based chatbot interface for PCLN."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        use_session_memory: bool = True,
        learn_on_chat: bool = False,
        learn_lr: float = 1e-5,
        save_on_exit: bool = True,
        decode_temperature: float = 0.75,
        decode_top_k: int = 40,
        decode_repetition_penalty: float = 1.25,
        decode_no_repeat_ngram_size: int = 3,
        decode_top_p: float = 0.9,
    ):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.use_session_memory = use_session_memory
        self.learn_on_chat = learn_on_chat
        self.learn_lr = learn_lr
        self.save_on_exit = save_on_exit
        self.decode_temperature = decode_temperature
        self.decode_top_k = decode_top_k
        self.decode_repetition_penalty = decode_repetition_penalty
        self.decode_no_repeat_ngram_size = decode_no_repeat_ngram_size
        self.decode_top_p = decode_top_p
        self.session_turns: list[tuple[str, str]] = []

        print(f"Device: {self.device}")
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        args = checkpoint["args"]
        self.args = args
        self.seq_len = args.seq_len

        self.vocab_mappings = checkpoint.get("vocab_mappings", None)
        self.tokenization = "word"
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}

        if self.vocab_mappings is None:
            print("WARNING: No vocab mappings in checkpoint.")
            print("   Train with real text data for text-based chat.")
            self.text_mode = False
        else:
            self.tokenization = self.vocab_mappings.get("tokenization", "word")
            if self.tokenization == "dummy":
                print("WARNING: Checkpoint was trained on the dummy dataset (random tokens).")
                print("   Use a WikiText or Shakespeare checkpoint for text chat.")
                self.text_mode = False
            elif self.tokenization == "char":
                self.char2id = self.vocab_mappings.get("char2id", {})
                raw_id2char = self.vocab_mappings.get("id2char", {})
                self.id2char = {int(k): v for k, v in raw_id2char.items()}
                self.text_mode = bool(self.char2id)
                print(f"[OK] Loaded char vocab: {len(self.char2id)} characters")
            else:
                self.word2id = self.vocab_mappings.get("word2id", {})
                raw_id2word = self.vocab_mappings.get("id2word", {})
                self.id2word = {int(k): v for k, v in raw_id2word.items()}
                self.text_mode = bool(self.word2id)
                print(f"[OK] Loaded word vocab: {len(self.word2id)} tokens")

        model_state = checkpoint["model_state"]
        embedding_weight = model_state["encoder.embedding.weight"]
        actual_vocab_size = embedding_weight.shape[0]
        self.vocab_size = actual_vocab_size
        self.unk_token_id = actual_vocab_size - 1
        print(f"Inferred vocab size from checkpoint: {actual_vocab_size}")

        self.model = build_pcln(args, actual_vocab_size, default_causal=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.word_codec: WordCodec | None = None
        self.char_codec: CharCodec | None = None
        if self.text_mode and self.tokenization == "char":
            self.char_codec = CharCodec(self.char2id, self.id2char, unk_id=0)
        elif self.text_mode:
            self.word_codec = WordCodec(self.word2id, self.id2word, unk_id=self.unk_token_id)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        if self.learn_on_chat:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_lr)
            print(f"[OK] Online learning enabled (lr={self.learn_lr})")

        if self.use_session_memory and args.use_memory:
            print("[OK] Session episodic memory: stores each turn in chat")
        print("Model loaded successfully!")

    def encode_text(self, text: str) -> list[int]:
        """Convert text to token IDs using the same rules as training."""
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")
        if self.char_codec is not None:
            return self.char_codec.encode(text)
        if self.word_codec is not None:
            return self.word_codec.encode(text)
        raise RuntimeError("No codec loaded")

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")
        if self.char_codec is not None:
            return self.char_codec.decode(token_ids)
        if self.word_codec is not None:
            return self.word_codec.decode(token_ids)
        raise RuntimeError("No codec loaded")

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        max_len: int = 50,
        temperature: float = 0.7,
        top_k: int | None = 40,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 0,
        top_p: float = 0.9,
        store_memory: bool = False,
    ) -> list[int]:
        """Generate tokens given a prompt."""
        if len(prompt_tokens) > self.seq_len:
            prompt_tokens = prompt_tokens[-self.seq_len :]

        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        generated: list[int] = []
        history = list(prompt_tokens)

        for _ in range(max_len):
            output = self.model(
                tokens,
                return_errors=False,
                store_memory=store_memory,
            )
            logits = output["logits"][:, -1, :]
            logits = apply_repetition_penalty(logits, history[-32:], repetition_penalty)
            logits = ban_repeating_ngrams(logits, history, no_repeat_ngram_size)
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tid = int(next_token.item())
            generated.append(tid)
            history.append(tid)
            tokens = torch.cat([tokens, next_token], dim=1)
            if tokens.size(1) > self.seq_len:
                tokens = tokens[:, -self.seq_len :]

        return generated

    def learn_from_turn(self, context_tokens: list[int], target_tokens: list[int]) -> float:
        """One gradient step on a conversation turn (optional online learning)."""
        if not self.learn_on_chat or self.optimizer is None:
            return 0.0
        if len(context_tokens) < 2 or len(target_tokens) < 1:
            return 0.0

        self.model.train()
        inp = torch.tensor([context_tokens], dtype=torch.long, device=self.device)
        tgt = torch.tensor([target_tokens], dtype=torch.long, device=self.device)

        output = self.model(inp, return_errors=True, store_memory=True)
        logits = output["logits"]
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )
        if output.get("errors"):
            for err in output["errors"]:
                loss = loss + 0.05 * (err ** 2).mean()
        loss = loss + 0.05 * output["load_balance_loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def respond(self, user_text: str, max_len: int = 40) -> str:
        """Generate a reply and optionally learn + remember the turn."""
        prompt_tokens = self.encode_text(user_text)
        if not prompt_tokens:
            return ""

        store = self.use_session_memory and self.args.use_memory
        generated = self.generate(
            prompt_tokens,
            max_len=max_len,
            temperature=self.decode_temperature,
            top_k=self.decode_top_k,
            repetition_penalty=self.decode_repetition_penalty,
            no_repeat_ngram_size=self.decode_no_repeat_ngram_size,
            top_p=self.decode_top_p,
            store_memory=store,
        )
        reply = self.decode_tokens(generated)

        if self.learn_on_chat:
            full_context = prompt_tokens + generated
            if len(full_context) > 1:
                inp = full_context[:-1][-self.seq_len:]
                tgt = full_context[1:][-self.seq_len:]
                loss = self.learn_from_turn(inp, tgt)
                if loss > 0:
                    print(f"  [learn] loss={loss:.4f}")

        self.session_turns.append((user_text, reply))
        return reply

    def save_checkpoint(self, path: Path | None = None) -> None:
        """Save model after online learning session."""
        path = path or self.checkpoint_path
        ck = {
            "epoch": 0,
            "step": 0,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "args": self.args,
            "vocab_mappings": self.vocab_mappings,
        }
        torch.save(ck, path)
        print(f"Saved checkpoint to {path}")

    def chat(self):
        """Run interactive text-based chat loop."""
        if not self.text_mode:
            print("\n" + "=" * 60)
            print("TOKEN MODE (Model from random/dummy data)")
            print("=" * 60)
            self._token_demo()
        else:
            print("\n" + "=" * 60)
            print("PCLN Text Chat")
            print("=" * 60)
            print(f"Tokenization: {self.tokenization}")
            mem = "on" if self.use_session_memory and self.args.use_memory else "off"
            learn = "on" if self.learn_on_chat else "off"
            print(f"Session memory: {mem} | Learn while chatting: {learn}")
            print("Commands: type a message, or quit/exit")
            print("=" * 60 + "\n")
            self._text_demo()

    def _text_demo(self):
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["quit", "exit"]:
                        break

                    print("\nModel: ", end="", flush=True)
                    reply = self.respond(user_input, max_len=40)
                    print(reply)
                    print()

                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    break
                except Exception as e:
                    print(f"Error: {e}\n")
        finally:
            if self.learn_on_chat and self.save_on_exit:
                self.save_checkpoint()
            print("Goodbye!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Text-based chat with PCLN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to best_model.pt (default: sprint wiki_stride64 if present)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling (1.0=off)")
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Ban repeating n-grams during generation (0=off)",
    )
    parser.add_argument(
        "--no-session-memory",
        action="store_true",
        help="Disable episodic memory writes during chat",
    )
    parser.add_argument(
        "--learn-on-chat",
        action="store_true",
        help="Update weights from each turn (slow, experimental)",
    )
    parser.add_argument("--learn-lr", type=float, default=1e-5)

    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint_path()
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        print("\nAvailable sprint checkpoints:")
        sprint = ROOT / "results" / "sprint"
        if sprint.exists():
            for pt in sorted(sprint.glob("*/best_model.pt")):
                print(f"  {pt}")
        return 1

    chatbot = ChatBot(
        args.checkpoint if args.checkpoint else str(checkpoint_path),
        device=args.device,
        use_session_memory=not args.no_session_memory,
        learn_on_chat=args.learn_on_chat,
        learn_lr=args.learn_lr,
        decode_temperature=args.temperature,
        decode_top_k=args.top_k,
        decode_repetition_penalty=args.repetition_penalty,
        decode_no_repeat_ngram_size=args.no_repeat_ngram_size,
        decode_top_p=args.top_p,
    )

    if not chatbot.text_mode:
        print(_chat_unavailable_message(checkpoint_path))
        return 1

    if args.prompt:
        prompt_tokens = chatbot.encode_text(args.prompt)
        generated = chatbot.generate(
            prompt_tokens,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_p=args.top_p,
            store_memory=chatbot.use_session_memory,
        )
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {chatbot.decode_tokens(generated)}")
    else:
        chatbot.chat()


if __name__ == "__main__":
    main()
