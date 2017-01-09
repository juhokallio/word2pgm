#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import unittest
import math
from scipy.special import gamma


def surface(r, n):
    numerator = 2 * math.pi ** (n * 0.5)
    denominator = gamma(0.5 * n)
    return (numerator / denominator) * r ** (n - 1)
    
def volume(r, n):
    numerator = math.pi ** (n * 0.5)
    denominator = gamma(0.5 * n + 1)
    return (numerator / denominator) * r ** n


class TestNPhere(unittest.TestCase):

    def test_volume(self):
        self.assertAlmostEqual(math.pi, volume(1, 2), delta=0.001,
                msg="Failed to calculate unit circle volume/surface")
        self.assertAlmostEqual(math.pi*4/3, volume(1, 3), delta=0.001,
                msg="Failed to calculate unit ball volume")

    def test_surface(self):
        self.assertAlmostEqual(2*math.pi, surface(1, 2), delta=0.001,
                msg="Failed to calculate unit circle surface")
        self.assertAlmostEqual(4*math.pi, surface(1, 3), delta=0.001,
                msg="Failed to calculate unit ball surface")
