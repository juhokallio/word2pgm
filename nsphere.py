#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import unittest
import math
from scipy.special import gamma, betainc


def surface(r, n):
    multiplier = (2 * math.pi ** (n * 0.5)) / gamma(0.5 * n)
    return multiplier * r ** (n - 1)
    
def volume(r, n):
    numerator = math.pi ** (n * 0.5)
    denominator = gamma(0.5 * n + 1)
    return (numerator / denominator) * r ** n

def cap(r, n, alpha):
    if alpha < 0.5 * math.pi:
        return 0.5 * surface(r, n) * betainc(0.5*(n-1), 0.5, math.sin(alpha)**2)
    else:
        non_surface = 0.5 * surface(r, n) * betainc(0.5*(n-1), 0.5, math.sin(math.pi-alpha)**2)
        return surface(r, n) - non_surface


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

    def test_cap_surface(self):
        self.assertAlmostEqual(0.5 * surface(1, 3), cap(1, 3, 0.5 * math.pi), delta=0.001,
            msg="Cap with alpha 0.5pi was not half the surface")
        self.assertAlmostEqual(surface(1, 3), cap(1, 3, math.pi), delta=0.001,
            msg="Cap with alpha pi was different from surface")
        self.assertEqual(0, cap(4, 5, 0),
            msg="Cap with alpha 0 was wrong")
