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

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        args = checkpoint["args"]
        self.seq_len = args.seq_len
        
        # Get vocab mappings from checkpoint
        self.vocab_mappings = checkpoint.get("vocab_mappings", None)
        
        if self.vocab_mappings is None:
            print("WARNING: No vocab mappings in checkpoint.")
            print("   The model was likely trained with dummy data (random tokens).")
            print("   For text-based chat, train with: --dataset wikitext2")
            self.text_mode = False
            # Create dummy vocab for fallback
            self.word2id = {}
            self.id2word = {}
        else:
            self.word2id = self.vocab_mappings.get("word2id", {})
            self.id2word = self.vocab_mappings.get("id2word", {})
            self.text_mode = True
            print(f"[OK] Loaded vocab: {len(self.word2id)} unique words")

        # Infer vocab size from model weights
        model_state = checkpoint["model_state"]
        embedding_weight = model_state["encoder.embedding.weight"]
        actual_vocab_size = embedding_weight.shape[0]
        self.vocab_size = actual_vocab_size
        self.unk_token_id = actual_vocab_size - 1  # Usually the last token
        print(f"Inferred vocab size from checkpoint: {actual_vocab_size}")

        # Build model with same config (including MoE if present)
        moe_kwargs = {}
        if hasattr(args, 'use_sparse_moe'):
            moe_kwargs['use_sparse_moe'] = args.use_sparse_moe
            moe_kwargs['num_experts'] = getattr(args, 'num_experts', 4)
            moe_kwargs['top_k_experts'] = getattr(args, 'top_k_experts', 2)

        self.model = PCLN(
            vocab_size=actual_vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_pcn_blocks=args.num_pcn_blocks,
            K_pcn=args.K_pcn,
            alpha_pcn=args.alpha_pcn,
            dropout=args.dropout,
            use_memory=args.use_memory,
            episodic_memory_size=args.episodic_memory_size,
            semantic_slots=args.semantic_slots,
            **moe_kwargs,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print("Model loaded successfully!")

    def encode_text(self, text: str) -> list[int]:
        """Convert text to token IDs.
        
        Args:
            text: input text string
        Returns:
            list of token IDs
        """
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")
        
        # Simple tokenization: split by spaces and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        
        tokens = []
        for word in words:
            token_id = self.word2id.get(word, self.unk_token_id)
            tokens.append(token_id)
        
        return tokens if tokens else [self.word2id.get(text[0], self.unk_token_id)]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text.
        
        Args:
            token_ids: list of token IDs
        Returns:
            decoded text string
        """
        if not self.text_mode:
            raise RuntimeError("Text mode not available - model trained on random tokens")
        
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
        """Generate tokens given a prompt.
        
        Args:
            prompt_tokens: list of token IDs to start from.
            max_len: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: use top-k sampling if set.
        Returns:
            generated token IDs (excluding the prompt).
        """
        # Prepare input - ensure it doesn't exceed seq_len
        prompt_len = len(prompt_tokens)
        if prompt_len > self.seq_len:
            prompt_tokens = prompt_tokens[-self.seq_len:]
        
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        generated = []
        for _ in range(max_len):
            # Forward pass
            output = self.model(tokens, return_errors=False)
            logits = output["logits"][:, -1, :]  # Last token logits (batch, vocab_size)

            # Apply temperature
            logits = logits / max(temperature, 0.01)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated.append(next_token.item())

            # Append to tokens for next iteration
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # If we exceed seq_len, drop the oldest tokens to keep context window
            if tokens.size(1) > self.seq_len:
                tokens = tokens[:, -self.seq_len:]

        return generated

    def chat(self):
        """Run interactive text-based chat loop."""
        if not self.text_mode:
            print("\n" + "=" * 60)
            print("TOKEN MODE (Model from random/dummy data)")
            print("=" * 60)
            print("\nThis model was trained on random tokens, not real text.")
            print("For text chat, train with: --dataset wikitext2")
            print("\nCommands:")
            print("  generate <num> - generate tokens")
            print("  quit/exit      - exit\n")
            self._token_demo()
        else:
            print("\n" + "=" * 60)
            print("PCLN Text Chat")
            print("=" * 60)
            print("\nTrained on Tiny Shakespeare - generates similar text")
            print("Commands:")
            print("  Your message - chat with model (press Enter)")
            print("  quit/exit    - exit")
            print("=" * 60 + "\n")
            self._text_demo()

    def _text_demo(self):
        """Run text-based demo."""
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                # Encode user text
                prompt_tokens = self.encode_text(user_input)
                
                if not prompt_tokens:
                    print("Could not parse input text.\n")
                    continue

                # Generate response
                print("\nModel: ", end="", flush=True)
                generated_tokens = self.generate(
                    prompt_tokens,
                    max_len=30,
                    temperature=0.9,
                    top_k=15,
                )

                # Decode back to text
                response_text = self.decode_tokens(generated_tokens)
                print(response_text)
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
        """Run token-based demo (for models without vocab)."""
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

                    # Generate from random prompt
                    prompt = [torch.randint(0, self.vocab_size, (1,)).item()]
                    generated = self.generate(
                        prompt,
                        max_len=max_len,
                        temperature=0.8,
                        top_k=10,
                    )

                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated}")
                    print(f"Length: {len(generated)}\n")
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
    main()
