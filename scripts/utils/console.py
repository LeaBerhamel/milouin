# -*- coding: utf-8 -*-
"""Helpers d'affichage console partagés par les scripts du projet."""

SEPARATOR_WIDTH = 60


def print_header(title: str) -> None:
    """Affiche un titre encadré par des séparateurs."""
    bar = "=" * SEPARATOR_WIDTH
    print(f"\n{bar}")
    print(title)
    print(f"{bar}\n")


def print_footer(title: str) -> None:
    """Affiche un titre de fin de section (même style que print_header, sans saut de ligne initial)."""
    bar = "=" * SEPARATOR_WIDTH
    print(f"{bar}")
    print(title)
    print(f"{bar}\n")
