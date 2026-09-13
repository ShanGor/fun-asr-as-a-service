import os
import unittest
from unittest.mock import patch

from funasr_core import config


class DeviceConfigTests(unittest.TestCase):
    def test_arm64_macos_defaults_to_mps(self):
        with patch.object(config.sys, "platform", "darwin"), \
             patch.object(config.platform, "machine", return_value="arm64"), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.load_config()["device"], "mps")

    def test_linux_default_stays_cuda(self):
        with patch.object(config.sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.load_config()["device"], "cuda")

    def test_explicit_device_wins(self):
        with patch.object(config.sys, "platform", "darwin"), \
             patch.object(config.platform, "machine", return_value="arm64"), \
             patch.dict(os.environ, {"FUNASR_DEVICE": "cpu"}, clear=True):
            self.assertEqual(config.load_config()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
