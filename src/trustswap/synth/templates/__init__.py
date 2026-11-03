"""Packaged Jinja2 confirmation families, with a fixed family/split manifest."""
from types import MappingProxyType

FAMILY_SPLITS = MappingProxyType({
    "label_value": "train",
    "two_column": "train",
    "dealer_caps": "train",
    "legal_preamble": "dev",
    "letter": "syn_test",
    "term_sheet": "syn_test",
})
