# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for zk_tax Python bindings."""

import pytest
from zk_tax import (
    Jurisdiction,
    FilingStatus,
    TaxBracket,
    find_bracket,
    get_brackets,
    list_jurisdictions,
    version,
)


class TestJurisdiction:
    """Test Jurisdiction enum."""

    def test_all_jurisdictions_exist(self):
        """Verify all 58 jurisdictions are accessible."""
        jurisdictions = [
            Jurisdiction.US, Jurisdiction.UK, Jurisdiction.DE, Jurisdiction.FR,
            Jurisdiction.CA, Jurisdiction.AU, Jurisdiction.JP, Jurisdiction.CN,
            Jurisdiction.IN, Jurisdiction.BR, Jurisdiction.MX, Jurisdiction.KR,
            Jurisdiction.IT, Jurisdiction.RU, Jurisdiction.TR, Jurisdiction.AR,
            Jurisdiction.ID, Jurisdiction.SA, Jurisdiction.ZA, Jurisdiction.CL,
            Jurisdiction.CO, Jurisdiction.PE, Jurisdiction.EC, Jurisdiction.UY,
            Jurisdiction.ES, Jurisdiction.NL, Jurisdiction.BE, Jurisdiction.AT,
            Jurisdiction.PT, Jurisdiction.IE, Jurisdiction.PL, Jurisdiction.SE,
            Jurisdiction.DK, Jurisdiction.FI, Jurisdiction.NO, Jurisdiction.CH,
            Jurisdiction.CZ, Jurisdiction.GR, Jurisdiction.HU, Jurisdiction.RO,
            Jurisdiction.UA, Jurisdiction.NZ, Jurisdiction.SG, Jurisdiction.HK,
            Jurisdiction.TW, Jurisdiction.MY, Jurisdiction.TH, Jurisdiction.VN,
            Jurisdiction.PH, Jurisdiction.PK, Jurisdiction.AE, Jurisdiction.IL,
            Jurisdiction.EG, Jurisdiction.QA, Jurisdiction.NG, Jurisdiction.KE,
            Jurisdiction.MA, Jurisdiction.GH,
        ]
        assert len(jurisdictions) == 58


class TestFilingStatus:
    """Test FilingStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all filing statuses are accessible."""
        statuses = [
            FilingStatus.SINGLE,
            FilingStatus.MARRIED_FILING_JOINTLY,
            FilingStatus.MARRIED_FILING_SEPARATELY,
            FilingStatus.HEAD_OF_HOUSEHOLD,
        ]
        assert len(statuses) == 4


class TestFindBracket:
    """Test find_bracket function."""

    def test_us_single_2024_low_income(self):
        """Test US single filer with low income."""
        bracket = find_bracket(15000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.index == 1  # 12% bracket
        assert bracket.rate_percent == 12.0

    def test_us_single_2024_mid_income(self):
        """Test US single filer with mid income."""
        bracket = find_bracket(85000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.index == 2  # 22% bracket
        assert bracket.rate_percent == 22.0

    def test_us_single_2024_high_income(self):
        """Test US single filer with high income."""
        bracket = find_bracket(600000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.index == 6  # 37% bracket
        assert bracket.rate_percent == 37.0

    def test_uk_2024(self):
        """Test UK brackets."""
        bracket = find_bracket(50000, Jurisdiction.UK, 2024, FilingStatus.SINGLE)
        assert bracket.rate_percent == 40.0  # Higher rate

    def test_singapore_low_tax(self):
        """Test Singapore's low tax rates."""
        bracket = find_bracket(20000, Jurisdiction.SG, 2024, FilingStatus.SINGLE)
        assert bracket.rate_percent <= 2.0

    def test_uae_zero_tax(self):
        """Test UAE has 0% personal income tax."""
        bracket = find_bracket(1000000, Jurisdiction.AE, 2024, FilingStatus.SINGLE)
        assert bracket.rate_percent == 0.0

    def test_bracket_contains_income(self):
        """Verify found bracket contains the income."""
        income = 75000
        bracket = find_bracket(income, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.contains(income)
        assert income >= bracket.lower
        assert income < bracket.upper or bracket.upper == 2**64 - 1

    def test_invalid_year_raises(self):
        """Test that invalid year raises ValueError."""
        with pytest.raises(ValueError):
            find_bracket(50000, Jurisdiction.US, 2019, FilingStatus.SINGLE)

        with pytest.raises(ValueError):
            find_bracket(50000, Jurisdiction.US, 2030, FilingStatus.SINGLE)


class TestGetBrackets:
    """Test get_brackets function."""

    def test_us_2024_has_7_brackets(self):
        """US has 7 federal tax brackets."""
        brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert len(brackets) == 7

    def test_brackets_sorted_by_lower(self):
        """Brackets should be sorted by lower bound."""
        brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
        for i in range(len(brackets) - 1):
            assert brackets[i].lower < brackets[i + 1].lower

    def test_brackets_progressive(self):
        """Rates should increase with higher brackets."""
        brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
        for i in range(len(brackets) - 1):
            assert brackets[i].rate_bps <= brackets[i + 1].rate_bps

    def test_first_bracket_starts_at_zero(self):
        """First bracket should start at 0."""
        brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert brackets[0].lower == 0

    def test_hungary_flat_tax(self):
        """Hungary has a flat 15% tax."""
        brackets = get_brackets(Jurisdiction.HU, 2024, FilingStatus.SINGLE)
        assert len(brackets) == 1
        assert brackets[0].rate_percent == 15.0


class TestListJurisdictions:
    """Test list_jurisdictions function."""

    def test_returns_58_jurisdictions(self):
        """Should return all 58 jurisdictions."""
        jurisdictions = list_jurisdictions()
        assert len(jurisdictions) == 58

    def test_returns_tuples(self):
        """Each item should be a (code, name) tuple."""
        jurisdictions = list_jurisdictions()
        for code, name in jurisdictions:
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) == 2
            assert len(name) > 0

    def test_us_present(self):
        """US should be in the list."""
        jurisdictions = list_jurisdictions()
        codes = [code for code, _ in jurisdictions]
        assert "US" in codes


class TestVersion:
    """Test version function."""

    def test_version_format(self):
        """Version should be semver-like."""
        v = version()
        assert isinstance(v, str)
        parts = v.split(".")
        assert len(parts) >= 2


class TestTaxBracket:
    """Test TaxBracket class."""

    def test_rate_percent_conversion(self):
        """rate_percent should be rate_bps / 100."""
        bracket = find_bracket(50000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.rate_percent == bracket.rate_bps / 100

    def test_contains_method(self):
        """contains() should work correctly."""
        bracket = find_bracket(50000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        assert bracket.contains(50000)
        assert bracket.contains(bracket.lower)
        if bracket.upper != 2**64 - 1:
            assert not bracket.contains(bracket.upper)

    def test_repr(self):
        """__repr__ should return a string."""
        bracket = find_bracket(50000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
        repr_str = repr(bracket)
        assert "TaxBracket" in repr_str
        assert "rate=" in repr_str


class TestMultipleYears:
    """Test bracket lookups across multiple years."""

    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024, 2025])
    def test_all_supported_years(self, year):
        """All years 2020-2025 should work."""
        bracket = find_bracket(50000, Jurisdiction.US, year, FilingStatus.SINGLE)
        assert bracket is not None
        assert bracket.rate_percent > 0


class TestMultipleJurisdictions:
    """Test bracket lookups across multiple jurisdictions."""

    @pytest.mark.parametrize("jurisdiction", [
        Jurisdiction.US, Jurisdiction.UK, Jurisdiction.DE, Jurisdiction.FR,
        Jurisdiction.JP, Jurisdiction.AU, Jurisdiction.CA, Jurisdiction.SG,
    ])
    def test_major_jurisdictions(self, jurisdiction):
        """Major jurisdictions should all work."""
        brackets = get_brackets(jurisdiction, 2024, FilingStatus.SINGLE)
        assert len(brackets) > 0
        assert all(b.rate_bps >= 0 for b in brackets)
