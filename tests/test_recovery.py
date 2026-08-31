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

    def test_signature_snapshot_isolated_from_nested_input_and_payload_mutation(self):
        config = {"batch_size": 32, "scoring": {"threshold": 0.5}}
        example_ids = ["visible:0001", "visible:0002"]
        signature = self._signature(
            effective_evaluation_config=config,
            expected_example_ids=example_ids,
        )
        expected = self._signature(
            effective_evaluation_config={"batch_size": 32, "scoring": {"threshold": 0.5}},
            expected_example_ids=["visible:0001", "visible:0002"],
        )

        config["scoring"]["threshold"] = 0.9
        example_ids.append("visible:0003")
        payload = signature.payload
        payload["seed"] = 42
        payload["effective_evaluation_config"]["scoring"]["threshold"] = 0.1

        self.assertEqual(signature.digest, expected.digest)
        self.assertEqual(signature.payload, expected.payload)
        self.assertIsNone(signature.first_difference(expected))


if __name__ == "__main__":
    unittest.main()
