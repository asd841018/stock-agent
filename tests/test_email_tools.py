import os
import unittest
from unittest.mock import patch

from src.tools.email import send_email_message, smtp_settings


class EmailToolTests(unittest.TestCase):
    def test_smtp_settings_defaults_to_gmail(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = smtp_settings()
        self.assertEqual(settings["host"], "smtp.gmail.com")
        self.assertEqual(settings["port"], 465)
        self.assertTrue(settings["use_ssl"])

    def test_send_email_message_returns_disabled_without_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            result = send_email_message("user@gmail.com", "Bullish", "hello")
        self.assertIn("Email sending is disabled", result)


if __name__ == "__main__":
    unittest.main()
