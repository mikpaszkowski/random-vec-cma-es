#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pakiet zawierający implementacje algorytmów CMA-ES.
"""

from algorithms.cma_es_standard import StandardCMAES
from algorithms.cma_es_modified import ModifiedCMAES

__all__ = ['StandardCMAES', 'ModifiedCMAES']
