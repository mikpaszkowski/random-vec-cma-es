#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł algorithms - zawiera implementacje algorytmów optymalizacji.
"""

from .cma_es_standard import StandardCMAES
from .cma_es_modified import ModifiedCMAES

__all__ = ['StandardCMAES', 'ModifiedCMAES']
