"""Smoke tests for the JARVIS local brain + server wiring (stdlib unittest)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis import brain  # noqa: E402


class TestBrain(unittest.TestCase):
    def test_greeting(self):
        self.assertIn("sir", brain.respond("hello").lower())

    def test_whoami(self):
        self.assertIn("JARVIS", brain.respond("who are you"))

    def test_time(self):
        self.assertIn("time", brain.respond("what time is it").lower())

    def test_help(self):
        self.assertIn("Here's what I can do", brain.respond("help"))

    def test_math(self):
        self.assertEqual(brain.evaluate_math("2+2")[1], "4")
        self.assertEqual(brain.evaluate_math("(12+8)*5")[1], "100")
        # safe: no eval/exec allowed
        self.assertFalse(brain.evaluate_math("__import__('os')")[0])

    def test_percent(self):
        self.assertEqual(brain.respond("what is 15% of 2400"), "360")

    def test_unit_conversion(self):
        r = brain.respond("convert 5 miles to km")
        self.assertIn("8.0467", r)
        r2 = brain.respond("how many kg in 200 pounds")
        self.assertIn("90.7185", r2)

    def test_temperature(self):
        self.assertEqual(brain.respond("convert 0 celsius to fahrenheit"),
                         "0 celsius = 32 fahrenheit")

    def test_joke(self):
        self.assertTrue(brain.respond("tell me a joke"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
