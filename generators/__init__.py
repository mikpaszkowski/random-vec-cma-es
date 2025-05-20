#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pakiet zawierający generatory liczb pseudolosowych.
"""

from generators.mersenne_twister import MersenneTwister
from generators.pcg import PCG

__all__ = ['MersenneTwister', 'PCG']


def get_generator(generator_type, seed=None):
    """
    Fabryka generatorów liczb pseudolosowych.
    
    Args:
        generator_type: Typ generatora ('mt' lub 'pcg').
        seed: Ziarno inicjalizacyjne dla generatora.
        
    Returns:
        Instancja żądanego generatora.
        
    Raises:
        ValueError: Jeśli podano nieznany typ generatora.
    """
    if generator_type.lower() == 'mt':
        return MersenneTwister(seed)
    elif generator_type.lower() == 'pcg':
        return PCG(seed)
    else:
        raise ValueError(f"Nieznany typ generatora: {generator_type}. "
                         f"Dostępne generatory: 'mt', 'pcg'.")
