"""Causal language-model constraints for PCLN.

The encoder must not let position t attend to tokens > t. Otherwise
full-sequence next-token loss is leaked and perplexity is not a real LM score.
"""

from __future__ import annotations

import unittest

import torch

from src.model import PCLN


class TestCausalLanguageModel(unittest.TestCase):
    def _tiny(self, **kwargs) -> PCLN:
        defaults = dict(
            vocab_size=32,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_pcn_blocks=1,
            K_pcn=1,
            use_memory=False,
            causal=True,
        )
        defaults.update(kwargs)
        return PCLN(**defaults)

    def test_past_logits_do_not_depend_on_future_tokens(self):
        torch.manual_seed(0)
        model = self._tiny()
        model.eval()
        a = torch.randint(0, 32, (2, 8))
        b = a.clone()
        b[:, -1] = (b[:, -1] + 1) % 32
        with torch.no_grad():
            la = model(a)["logits"]
            lb = model(b)["logits"]
        self.assertTrue(torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5, rtol=1e-5))

    def test_last_position_does_change_when_last_token_changes(self):
        torch.manual_seed(1)
        model = self._tiny()
        model.eval()
        a = torch.randint(0, 32, (1, 8))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 3) % 32
        with torch.no_grad():
            la = model(a)["logits"]
            lb = model(b)["logits"]
        self.assertFalse(torch.allclose(la[:, -1], lb[:, -1], atol=1e-5, rtol=1e-5))

    def test_non_causal_flag_still_loads_bidirectional_path(self):
        torch.manual_seed(2)
        model = self._tiny(causal=False)
        model.eval()
        a = torch.randint(0, 32, (1, 8))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % 32
        with torch.no_grad():
            la = model(a)["logits"]
            lb = model(b)["logits"]
        # Bidirectional models may let the last token affect earlier logits.
        self.assertEqual(la.shape, lb.shape)


if __name__ == "__main__":
    unittest.main()
