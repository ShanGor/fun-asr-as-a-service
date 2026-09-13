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

    def test_https_settings_are_loaded(self):
        env = {
            "FUNASR_SSL_CERTFILE": "/tmp/cert.pem",
            "FUNASR_SSL_KEYFILE": "/tmp/key.pem",
            "FUNASR_SSL_KEYFILE_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()
        self.assertEqual(loaded["ssl_certfile"], "/tmp/cert.pem")
        self.assertEqual(loaded["ssl_keyfile"], "/tmp/key.pem")
        self.assertEqual(loaded["ssl_keyfile_password"], "secret")


if __name__ == "__main__":
    unittest.main()
