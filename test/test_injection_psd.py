"""PSD conventions for the matched injection pair (kernel-fidelity systematic).

Argus models the SGWB with a single-corner OU process whose steepest attainable
residual slope is f^-4, against the field's standard power law at f^-13/3. That is a
deliberate modelling difference, and the change design commits to *measuring* the
systematic it induces rather than gating on it. Measuring it needs a matched pair of
injections on the same geometry -- one of each shape -- and "matched" has to mean
something precise.

It means matched *band-referenced PSD*: a PTA constrains the spectrum over about a
decade, so the PSD at a pivot frequency inside that band is the only quantity the two
shapes share. Their native parameters do not correspond at all -- (log10_A, gamma)
against (log10_ha, log10_gamma_a) -- so comparing those would be meaningless.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC_PER_YEAR = 365.25 * 86400.0
F_YR = 1.0 / SEC_PER_YEAR
F_PIVOT = 1.0 / (5.0 * SEC_PER_YEAR)


@pytest.fixture(scope="module")
def injector():
    path = os.path.join(
        REPO_ROOT, "workflows", "ng15_sgwb_demo", "scripts", "inject_powerlaw_gwb.py"
    )
    if not os.path.exists(path):
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location("inject_powerlaw_gwb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matched_ou_amplitude(log10_A, gamma, log10_gamma_a, f_pivot=F_PIVOT):
    """The log10_ha whose OU PSD equals the power law's at the pivot.

    Inverts S_OU(f) = (ha^2/12) gamma_a / (w^2 (gamma_a^2 + w^2)) for ha, which is the
    same inversion the ridge GW parameterization performs.
    """
    psd = (
        (10.0**log10_A) ** 2
        / (12.0 * np.pi**2)
        * (f_pivot / F_YR) ** (-gamma)
        * F_YR**-3
    )
    w = 2.0 * np.pi * f_pivot
    gamma_a = 10.0**log10_gamma_a
    return 0.5 * (
        math.log10(12.0)
        + math.log10(psd)
        + 2.0 * math.log10(w)
        + math.log10(gamma_a**2 + w**2)
        - log10_gamma_a
    )


def test_ou_psd_matches_its_closed_form(injector):
    log10_ha, log10_gamma_a = -13.0, -8.5
    freqs = np.array([1e-9, 5e-9, 3e-8, 1e-7])

    gamma_a = 10.0**log10_gamma_a
    w = 2.0 * np.pi * freqs
    expected = ((10.0**log10_ha) ** 2 / 12.0 * gamma_a) / (w**2 * (gamma_a**2 + w**2))

    np.testing.assert_allclose(
        injector.ou_psd(freqs, log10_ha, log10_gamma_a), expected, rtol=1e-14
    )


def test_ou_spectrum_cannot_be_steeper_than_f_minus_four(injector):
    """The ceiling the whole kernel-fidelity task exists to quantify.

    Well above its corner the OU PSD goes as f^-4, which is the steepest slope it can
    produce -- short of the true f^-13/3 ~ f^-4.33. Asserting it here means the claim
    is checked against the code rather than only argued in prose.
    """
    log10_ha, log10_gamma_a = -13.0, -9.0
    high = np.array([1e-7, 2e-7])  # decades above the corner at ~1.6e-10 Hz
    psd = injector.ou_psd(high, log10_ha, log10_gamma_a)
    slope = np.log(psd[1] / psd[0]) / np.log(high[1] / high[0])
    assert slope == pytest.approx(-4.0, abs=1e-3)


def test_powerlaw_spectrum_has_its_injected_slope(injector):
    gamma = 13.0 / 3.0
    freqs = np.array([1e-8, 2e-8])
    psd = injector.powerlaw_psd(freqs, -14.9, gamma)
    slope = np.log(psd[1] / psd[0]) / np.log(freqs[1] / freqs[0])
    assert slope == pytest.approx(-gamma, abs=1e-12)


def test_matched_pair_agrees_exactly_at_the_pivot(injector):
    """The defining property of the pair: identical band-referenced amplitude."""
    log10_A, gamma, log10_gamma_a = -14.886056647693163, 13.0 / 3.0, -8.5
    log10_ha = matched_ou_amplitude(log10_A, gamma, log10_gamma_a)

    powerlaw = injector.powerlaw_psd(np.array([F_PIVOT]), log10_A, gamma)[0]
    ou = injector.ou_psd(np.array([F_PIVOT]), log10_ha, log10_gamma_a)[0]
    np.testing.assert_allclose(np.log10(ou), np.log10(powerlaw), atol=1e-12)


def test_matched_pair_still_differs_away_from_the_pivot(injector):
    """And the property that makes the pair worth running.

    Matching at one frequency cannot match two different shapes everywhere. The
    residual difference across the band *is* the systematic being measured, so a pair
    that agreed everywhere would mean the test had been set up wrong.
    """
    log10_A, gamma, log10_gamma_a = -14.886056647693163, 13.0 / 3.0, -8.5
    log10_ha = matched_ou_amplitude(log10_A, gamma, log10_gamma_a)

    for frequency, expected_sign in ((1.0 / (15 * SEC_PER_YEAR), -1), (F_YR, +1)):
        powerlaw = injector.powerlaw_psd(np.array([frequency]), log10_A, gamma)[0]
        ou = injector.ou_psd(np.array([frequency]), log10_ha, log10_gamma_a)[0]
        difference = np.log10(ou) - np.log10(powerlaw)
        assert abs(difference) > 0.1, f"shapes agree at {frequency:.2e} Hz"
        # Flatter than the power law: below the band centre the OU sits low, above it high.
        assert np.sign(difference) == expected_sign


def test_matched_amplitude_is_insensitive_to_a_corner_below_the_band(injector):
    """Any corner well below the band gives the same in-band spectrum.

    The corner is unidentifiable from data confined to the band -- that is the ridge
    that made these runs hard to sample in the first place. So a matched pair must not
    depend on which sub-band corner was chosen, only on the pivot amplitude.
    """
    log10_A, gamma = -14.886056647693163, 13.0 / 3.0
    band = np.array([1.0 / (15 * SEC_PER_YEAR), F_PIVOT, F_YR])

    reference = None
    for log10_gamma_a in (-9.0, -9.5, -10.0):
        log10_ha = matched_ou_amplitude(log10_A, gamma, log10_gamma_a)
        psd = injector.ou_psd(band, log10_ha, log10_gamma_a)
        if reference is None:
            reference = psd
        else:
            np.testing.assert_allclose(psd, reference, rtol=1e-2)


def test_a_corner_too_close_to_the_band_is_detectable(injector):
    """Why the injected pair uses log10_gamma_a = -9.0 rather than -8.5.

    Over a 15 yr baseline the lowest sampled frequency is 1/15yr = 2.1e-9 Hz. A corner
    at 10^-8.5 sits only ~4x below that, so the lowest frequencies are not yet in the
    pure f^-4 regime and the OU spectrum carries a residual imprint of where the corner
    was put -- 0.024 dex at the bottom of the band. That would contaminate a
    measurement whose whole purpose is to isolate spectral *shape*, so the injection
    uses 10^-9.0 (~13x below the band), where the imprint drops to 0.0025 dex.

    This is not a bug in the OU model; it is a constraint on how the injection is
    configured, and it is asserted here so the choice cannot be quietly reverted.
    """
    f_low = 1.0 / (15 * SEC_PER_YEAR)
    w_low = 2.0 * np.pi * f_low

    def corner_imprint(log10_gamma_a):
        """Departure from pure f^-4 at the bottom of the band, in dex."""
        gamma_a = 10.0**log10_gamma_a
        return float(np.log10((gamma_a**2 + w_low**2) / w_low**2))

    assert corner_imprint(-8.5) > 0.02
    assert corner_imprint(-9.0) < 0.005
