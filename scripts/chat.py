"""Interactive text-based chat with PCLN.

Load a trained model and chat in natural language.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PCLN


class ChatBot:
    """Text-based chatbot interface for PCLN."""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Device: {self.device}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        args = checkpoint["args"]
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
        temperature: float = 0.8,
        top_k: int | None = None,
    ) -> list[int]:
        """Generate tokens given a prompt."""
        if len(prompt_tokens) > self.seq_len:
            prompt_tokens = prompt_tokens[-self.seq_len :]

        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        generated = []

        for _ in range(max_len):
            output = self.model(tokens, return_errors=False)
            logits = output["logits"][:, -1, :]
            logits = logits / max(temperature, 0.01)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token], dim=1)

            if tokens.size(1) > self.seq_len:
                tokens = tokens[:, -self.seq_len :]

        return generated

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
            print("Commands: type a message, or quit/exit")
            print("=" * 60 + "\n")
            self._text_demo()

    def _text_demo(self):
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                prompt_tokens = self.encode_text(user_input)
                if not prompt_tokens:
                    print("Could not parse input text.\n")
                    continue

                print("\nModel: ", end="", flush=True)
                generated_tokens = self.generate(
                    prompt_tokens,
                    max_len=30,
                    temperature=0.9,
                    top_k=15,
                )
                print(self.decode_tokens(generated_tokens))
                print()

            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    def _token_demo(self):
        while True:
            try:
                command = input("Enter command or 'quit' to exit: ").strip()
                if not command:
                    continue
                if command.lower() in ["quit", "exit"]:
                    print("Exiting...")
                    break

                if command.lower().startswith("generate"):
                    parts = command.split()
                    try:
                        max_len = int(parts[1]) if len(parts) > 1 else 20
                    except (ValueError, IndexError):
                        max_len = 20

                    prompt = [torch.randint(0, self.vocab_size, (1,)).item()]
                    generated = self.generate(prompt, max_len=max_len, temperature=0.8, top_k=10)
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated}\n")
                else:
                    print("Unknown command. Try 'generate <num>' or 'quit'.\n")

            except EOFError:
                print("\nExiting...")
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Text-based chat with PCLN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        print("Train a model first with: python scripts/train_full.py")
        return

    chatbot = ChatBot(args.checkpoint, device=args.device)
    chatbot.chat()


if __name__ == "__main__":
    main()
