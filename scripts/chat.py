"""Interactive text-based chat with PCLN.

Load a trained model and chat in natural language.
Supports session episodic memory, repetition penalty, and optional online learning.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PCLN


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: list[int],
    penalty: float,
) -> torch.Tensor:
    """Reduce logits for tokens that appeared recently (HF-style repetition penalty)."""
    if penalty <= 0 or not token_history:
        return logits
    out = logits.clone()
    for tid in set(token_history):
        score = out[:, tid]
        out[:, tid] = torch.where(score > 0, score / penalty, score * penalty)
    return out


def _ban_repeating_ngrams(
    logits: torch.Tensor,
    token_history: list[int],
    ngram_size: int,
) -> torch.Tensor:
    """Block tokens that would repeat an n-gram already seen in history."""
    if ngram_size <= 1 or len(token_history) < ngram_size - 1:
        return logits
    out = logits.clone()
    prefix = tuple(token_history[-(ngram_size - 1):])
    banned: set[int] = set()
    for i in range(len(token_history) - ngram_size + 1):
        if tuple(token_history[i : i + ngram_size - 1]) == prefix:
            banned.add(token_history[i + ngram_size - 1])
    for tid in banned:
        out[:, tid] = float("-inf")
    return out


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
    ):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.use_session_memory = use_session_memory
        self.learn_on_chat = learn_on_chat
        self.learn_lr = learn_lr
        self.save_on_exit = save_on_exit
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
            if self.tokenization == "char":
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

        model_kwargs = {
            "vocab_size": actual_vocab_size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "num_pcn_blocks": args.num_pcn_blocks,
            "K_pcn": args.K_pcn,
            "alpha_pcn": args.alpha_pcn,
            "dropout": args.dropout,
            "use_memory": args.use_memory,
            "episodic_memory_size": args.episodic_memory_size,
            "semantic_slots": args.semantic_slots,
            "use_sparse_moe": getattr(args, "use_sparse_moe", False),
            "use_dynamic_neurons": getattr(args, "use_dynamic_neurons", False),
            "num_experts": getattr(args, "num_experts", 4),
            "top_k_experts": getattr(args, "top_k_experts", 2),
            "num_neurons": getattr(args, "num_neurons", 256),
            "top_k_neurons": getattr(args, "top_k_neurons", 32),
        }

        self.model = PCLN(**model_kwargs).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        if self.learn_on_chat:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_lr)
            print(f"[OK] Online learning enabled (lr={self.learn_lr})")

        if self.use_session_memory and args.use_memory:
            print("[OK] Session episodic memory: stores each turn in chat")
        print("Model loaded successfully!")

    def encode_text(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")

        if self.tokenization == "char":
            tokens = [self.char2id.get(ch, 0) for ch in text]
            return tokens if tokens else [0]

        words = re.findall(r"\b\w+\b", text.lower())
        tokens = [self.word2id.get(word, self.unk_token_id) for word in words]
        return tokens if tokens else [self.unk_token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")

        if self.tokenization == "char":
            return "".join(self.id2char.get(token_id, "") for token_id in token_ids)

        words = []
        for token_id in token_ids:
            word = self.id2word.get(token_id, "<unk>")
            if word != "<unk>":
                words.append(word)
        return " ".join(words)

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        max_len: int = 50,
        temperature: float = 0.7,
        top_k: int | None = 40,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 0,
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
            logits = _apply_repetition_penalty(logits, history[-32:], repetition_penalty)
            logits = _ban_repeating_ngrams(logits, history, no_repeat_ngram_size)
            logits = logits / max(temperature, 0.01)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tid = next_token.item()
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
            temperature=0.75,
            top_k=40,
            repetition_penalty=1.25,
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
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return

    chatbot = ChatBot(
        args.checkpoint,
        device=args.device,
        use_session_memory=not args.no_session_memory,
        learn_on_chat=args.learn_on_chat,
        learn_lr=args.learn_lr,
    )

    if args.prompt:
        prompt_tokens = chatbot.encode_text(args.prompt)
        generated = chatbot.generate(
            prompt_tokens,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            store_memory=chatbot.use_session_memory,
        )
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {chatbot.decode_tokens(generated)}")
    else:
        chatbot.chat()


if __name__ == "__main__":
    main()
