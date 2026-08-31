import unittest

from runner.recovery import StageSignature


class StageSignatureTests(unittest.TestCase):
    def _signature(self, **changes):
        values = {
            "manifest_digest": "sha256:manifest", "protocol_version": "phase1-2026-08-29",
            "upstream_commit": "a" * 40, "upstream_tree": "b" * 40,
            "model_revision": "c" * 40, "seed": 17, "stage": "evaluation",
            "epoch": 1, "checkpoint_digest": "sha256:checkpoint",
            "effective_evaluation_config": {"batch_size": 32},
            "expected_example_ids": ["visible:0001", "visible:0002"],
        }
        values.update(changes)
        return StageSignature.create(**values)

    def test_equal_inputs_produce_equal_canonical_digest(self):
        self.assertEqual(self._signature().digest, self._signature().digest)

    def test_first_difference_names_changed_signature_field(self):
        self.assertEqual(self._signature().first_difference(self._signature(seed=42)), "seed")


if __name__ == "__main__":
    unittest.main()
