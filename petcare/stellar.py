"""Stellar parameters, density, and limb darkening.

Parameters come from, in order of preference:

1. a **project catalogue** (parquet/CSV) given by ``--catalog`` or
   ``$PETCARE_STELLAR_CATALOG``.  Columns are looked up under several names so
   that the catalogue can gain better parameters without this module changing:
   a plain ``rad``/``mass``/``teff`` wins over the TIC's ``tic_rad`` etc.;
2. the **TIC** at MAST, cached;
3. nothing, in which case every consumer must degrade to INCONCLUSIVE rather
   than substituting a solar default silently.

Young-star parameters are poor, and the physical tests here are built around
that: :func:`reconcile_density` reports *what it would take* to reconcile a
discrepant transit duration rather than vetoing the candidate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .cache import cached, ensure_certificates, with_retries

ENV_CATALOG = "PETCARE_STELLAR_CATALOG"

RSUN_M = 6.957e8
MSUN_KG = 1.98892e30
RHO_SUN_CGS = 1.408          # g/cm^3
G_SI = 6.67430e-11
DAY_S = 86400.0

# Column aliases, best first.  Our target catalogue currently carries the TIC
# values under a tic_ prefix and will gain fitted ones; both are handled.
_ALIASES = {
    "radius": ["rad", "radius", "st_rad", "R_star", "rstar", "tic_rad"],
    "e_radius": ["e_rad", "rad_err", "st_raderr1", "tic_e_rad"],
    "mass": ["mass", "st_mass", "M_star", "mstar", "tic_mass"],
    "e_mass": ["e_mass", "mass_err", "st_masserr1", "tic_e_mass"],
    "teff": ["teff", "Teff", "st_teff", "tic_Teff"],
    "e_teff": ["e_Teff", "teff_err", "st_tefferr1", "tic_e_Teff"],
    "logg": ["logg", "st_logg", "tic_logg"],
    "e_logg": ["e_logg", "tic_e_logg"],
    "rho": ["rho", "st_dens", "tic_rho"],
    "e_rho": ["e_rho", "tic_e_rho"],
    "tmag": ["T_final", "Tmag", "tmag", "sy_tmag"],
    "ra": ["ra", "RA", "ra_deg"],
    "dec": ["dec", "Dec", "DEC", "dec_deg"],
    "age_myr": ["age_myr", "age"],
    "e_age_lo": ["age_lo_myr"],
    "e_age_hi": ["age_hi_myr"],
    "gaia_dr3": ["gaia_dr3_source_id", "gaia_dr3", "source_id"],
    "dist_pc": ["dist_pc", "tic_d", "sy_dist"],
    "ruwe": ["ruwe"],
    "contratio": ["contratio"],
    "vsini": ["vsini", "st_vsin", "vbroad"],
    "prot_days": ["prot_days", "prot"],
}


@dataclass
class Star:
    """Everything the physical tests need, plus where it came from."""

    tic: int | None = None
    gaia_dr3: int | None = None
    name: str | None = None
    ra: float | None = None
    dec: float | None = None
    tmag: float | None = None

    radius: float | None = None          # R_sun
    e_radius: float | None = None
    mass: float | None = None            # M_sun
    e_mass: float | None = None
    teff: float | None = None            # K
    e_teff: float | None = None
    logg: float | None = None
    e_logg: float | None = None
    rho: float | None = None             # g/cm^3
    e_rho: float | None = None

    age_myr: float | None = None
    e_age_lo: float | None = None
    e_age_hi: float | None = None
    dist_pc: float | None = None
    ruwe: float | None = None
    contratio: float | None = None
    vsini: float | None = None
    prot_days: float | None = None

    source: str = "none"
    source_detail: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------------

    def density(self) -> tuple[float | None, float | None]:
        """Mean stellar density in g/cm^3, from M/R^3 when both are known.

        A catalogue ``rho`` is used only if mass and radius are unavailable:
        the TIC's rho is itself derived, and mixing two derivations of the
        same quantity is how two depth estimators came to disagree by a median
        factor of 0.68 in an earlier analysis.
        """
        if self.mass and self.radius and self.radius > 0:
            rho = RHO_SUN_CGS * self.mass / self.radius ** 3
            frac = 0.0
            if self.e_mass and self.mass:
                frac += (self.e_mass / self.mass) ** 2
            if self.e_radius and self.radius:
                frac += (3 * self.e_radius / self.radius) ** 2
            return float(rho), float(rho * np.sqrt(frac)) if frac else None
        if self.rho:
            return float(self.rho), (float(self.e_rho) if self.e_rho else None)
        return None, None

    def has_physical_parameters(self) -> bool:
        return self.density()[0] is not None

    def limb_darkening(self) -> tuple[float, float, str]:
        return quadratic_ld(self.teff, self.logg)

    def to_dict(self) -> dict:
        d = asdict(self)
        rho, e_rho = self.density()
        d["density_gcc"] = rho
        d["e_density_gcc"] = e_rho
        u1, u2, ld_src = self.limb_darkening()
        d["ld_u1"], d["ld_u2"], d["ld_source"] = u1, u2, ld_src
        return d

    def describe(self) -> str:
        rho, _ = self.density()
        bits = [self.name or (f"TIC {self.tic}" if self.tic else "?")]
        if self.teff:
            bits.append(f"Teff={self.teff:.0f}K")
        if self.radius:
            bits.append(f"R={self.radius:.3f}Rsun")
        if self.mass:
            bits.append(f"M={self.mass:.3f}Msun")
        if rho:
            bits.append(f"rho={rho:.3f}g/cc")
        if self.age_myr:
            bits.append(f"age={self.age_myr:.1f}Myr")
        bits.append(f"[{self.source}]")
        return "  ".join(bits)


# --------------------------------------------------------------------------
# catalogue lookup


def catalog_path(explicit: str | None = None) -> Path | None:
    p = explicit or os.environ.get(ENV_CATALOG)
    return Path(p).expanduser() if p else None


def _pick(row, key):
    for name in _ALIASES.get(key, [key]):
        if name in row and row[name] is not None:
            v = row[name]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                return fv
    return None


def _pick_id(row, key):
    """Like _pick, for an identifier: never through float()."""
    for name in _ALIASES.get(key, [key]):
        v = _int(row.get(name))
        if v is not None:
            return v
    return None


def from_catalog(tic: int | None = None, gaia_dr3: int | None = None,
                 path: str | None = None) -> Star | None:
    """Look a star up in the project catalogue.  Returns None if absent."""
    cp = catalog_path(path)
    if cp is None or not cp.exists():
        return None

    import pandas as pd

    df = _load_catalog(str(cp))
    row = None
    if tic is not None and "tic_id" in df.columns:
        # tic_id is a string column in our target catalogue; compare as strings
        # so that 166527623 and "166527623" both hit.
        m = df["tic_id"].astype(str).str.strip() == str(int(tic))
        if m.any():
            row = df[m].iloc[0]
    if row is None and gaia_dr3 is not None:
        for col in ("gaia_dr3_source_id", "source_id"):
            if col in df.columns:
                m = df[col].astype(str).str.strip() == str(int(gaia_dr3))
                if m.any():
                    row = df[m].iloc[0]
                    break
    if row is None:
        return None

    def _scalar_or_none(v):
        # some catalogue columns hold arrays (e.g. `sectors`); pd.isna on one
        # of those returns an array, which is not a truth value
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        try:
            return None if pd.isna(v) else v
        except (TypeError, ValueError):
            return v

    d = {k: _scalar_or_none(v) for k, v in row.to_dict().items()}
    star = Star(
        tic=int(tic) if tic is not None else _int(d.get("tic_id")),
        gaia_dr3=_pick_id(d, "gaia_dr3"),
        ra=_pick(d, "ra"), dec=_pick(d, "dec"), tmag=_pick(d, "tmag"),
        radius=_pick(d, "radius"), e_radius=_pick(d, "e_radius"),
        mass=_pick(d, "mass"), e_mass=_pick(d, "e_mass"),
        teff=_pick(d, "teff"), e_teff=_pick(d, "e_teff"),
        logg=_pick(d, "logg"), e_logg=_pick(d, "e_logg"),
        rho=_pick(d, "rho"), e_rho=_pick(d, "e_rho"),
        age_myr=_pick(d, "age_myr"), e_age_lo=_pick(d, "e_age_lo"),
        e_age_hi=_pick(d, "e_age_hi"), dist_pc=_pick(d, "dist_pc"),
        ruwe=_pick(d, "ruwe"), contratio=_pick(d, "contratio"),
        vsini=_pick(d, "vsini"), prot_days=_pick(d, "prot_days"),
        source=f"catalog:{cp.name}",
    )
    star.source_detail = {
        "catalog": str(cp),
        "primary_region": d.get("primary_region"),
        "age_ref": d.get("age_ref"),
        "age_method": d.get("age_method"),
        "n_neighbours": d.get("n_neighbours"),
        "brightest_contaminant_flux_fraction": d.get("brightest_contaminant_flux_fraction"),
        "columns_used": {k: next((n for n in _ALIASES[k] if n in d and d[n] is not None), None)
                         for k in ("radius", "mass", "teff", "logg")},
    }
    if star.name is None and star.tic:
        star.name = f"TIC {star.tic}"
    return star


_CATALOG_CACHE: dict[str, object] = {}


def _load_catalog(path: str):
    """Read and memoise the catalogue -- it is 30k rows and gets hit per target."""
    if path in _CATALOG_CACHE:
        return _CATALOG_CACHE[path]
    import pandas as pd

    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    _CATALOG_CACHE[path] = df
    return df


# --------------------------------------------------------------------------
# TIC fallback


def from_tic(tic: int) -> Star | None:
    """Query the TIC at MAST.  Cached; returns None if the query fails."""
    ensure_certificates()

    def _query():
        from astroquery.mast import Catalogs
        return Catalogs.query_criteria(catalog="TIC", ID=int(tic))

    try:
        tbl = cached("tic", "row", int(tic))(
            lambda: with_retries(_query, what=f"TIC {tic}"))
    except Exception:  # noqa: BLE001 - absence of a catalogue row is normal
        return None
    if tbl is None or len(tbl) == 0:
        return None
    r = {c: (None if np.ma.is_masked(tbl[c][0]) else tbl[c][0]) for c in tbl.colnames}

    star = Star(
        tic=int(tic), ra=_f(r.get("ra")), dec=_f(r.get("dec")),
        tmag=_f(r.get("Tmag")), radius=_f(r.get("rad")), e_radius=_f(r.get("e_rad")),
        mass=_f(r.get("mass")), e_mass=_f(r.get("e_mass")),
        teff=_f(r.get("Teff")), e_teff=_f(r.get("e_Teff")),
        logg=_f(r.get("logg")), e_logg=_f(r.get("e_logg")),
        rho=_f(r.get("rho")), e_rho=_f(r.get("e_rho")),
        dist_pc=_f(r.get("d")), contratio=_f(r.get("contratio")),
        gaia_dr3=_int(r.get("GAIA")), source="TIC",
    )
    star.source_detail = {"tic_version": str(r.get("version")),
                          "objType": str(r.get("objType"))}
    star.name = f"TIC {tic}"
    return star


#: Fractional R* uncertainty assumed when the catalogue states none.  The TIC
#: carries no radius error for many distant hot stars, and
#: without one the radius tests could only flag.  10%:
#: comparable to the TIC-vs-literature scatter for the calibration hosts.
ASSUMED_RADIUS_FRAC_ERR = 0.10


def assume_radius_error(star: Star) -> Star:
    """Give a radius with no stated error an assumed one, and record that."""
    if star.radius and not star.e_radius:
        star.e_radius = ASSUMED_RADIUS_FRAC_ERR * star.radius
        star.source_detail["e_radius_assumed_frac"] = ASSUMED_RADIUS_FRAC_ERR
    return star


def radius_error_note(star: Star) -> str:
    """' (assumed 10%)' when the R* error is ours rather than the catalogue's."""
    f = star.source_detail.get("e_radius_assumed_frac")
    return f" (error assumed {f:.0%}; the catalogue states none)" if f else ""


def get_star(tic: int | None = None, gaia_dr3: int | None = None,
             catalog: str | None = None, allow_tic: bool = True) -> Star:
    """Best available parameters, filling catalogue gaps from the TIC.

    Always returns a :class:`Star`; check :meth:`Star.has_physical_parameters`
    before using it for anything physical.  A star with no parameters is a
    normal outcome at 30k targets, not an error.
    """
    star = from_catalog(tic=tic, gaia_dr3=gaia_dr3, path=catalog)
    if star is not None and star.has_physical_parameters():
        return star

    tic_star = from_tic(tic) if (allow_tic and tic is not None) else None
    if star is None:
        return tic_star or Star(tic=tic, gaia_dr3=gaia_dr3,
                                name=f"TIC {tic}" if tic else None)

    if tic_star is not None:
        filled = []
        for fld in ("radius", "e_radius", "mass", "e_mass", "teff", "e_teff",
                    "logg", "e_logg", "rho", "e_rho", "ra", "dec", "tmag",
                    "dist_pc", "contratio"):
            if getattr(star, fld) is None and getattr(tic_star, fld) is not None:
                setattr(star, fld, getattr(tic_star, fld))
                filled.append(fld)
        if filled:
            star.source += "+TIC"
            star.source_detail["filled_from_tic"] = filled
    return star


# --------------------------------------------------------------------------
# limb darkening


# Approximate quadratic limb-darkening coefficients for the TESS band at
# solar metallicity and logg 4.5, following the shape of the Claret (2017)
# tables.  Coarse on purpose: the vetting statistics here (depth ratios,
# duration, odd/even) move by well under their own uncertainties across this
# range, and the alternative -- a hard dependency on an LD package and a
# network fetch per target -- is not worth it at 30k stars.  Do not use these
# for a published radius.
_LD_TEFF = np.array([3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], float)
_LD_U1 = np.array([0.28, 0.25, 0.34, 0.42, 0.44, 0.42, 0.36, 0.31, 0.27])
_LD_U2 = np.array([0.40, 0.40, 0.28, 0.22, 0.20, 0.21, 0.24, 0.26, 0.28])

#: Used when Teff is unknown.  Mid-K, the modal host in this sample.
LD_DEFAULT = (0.42, 0.22)


def quadratic_ld(teff: float | None, logg: float | None = None) -> tuple[float, float, str]:
    """``(u1, u2, source)`` for the TESS band.

    ``logg`` is accepted and ignored: over 4.0-5.0 the coefficients move by
    less than the spread already implied by the coarse Teff grid.
    """
    if teff is None or not np.isfinite(teff):
        return LD_DEFAULT[0], LD_DEFAULT[1], "default (Teff unknown)"
    t = float(np.clip(teff, _LD_TEFF[0], _LD_TEFF[-1]))
    u1 = float(np.interp(t, _LD_TEFF, _LD_U1))
    u2 = float(np.interp(t, _LD_TEFF, _LD_U2))
    note = "approx Claret(2017) TESS grid"
    if not (_LD_TEFF[0] <= (teff or 0) <= _LD_TEFF[-1]):
        note += f" (Teff {teff:.0f} K clipped to grid)"
    return u1, u2, note


# --------------------------------------------------------------------------
# transit geometry


def a_over_rs(period_days: float, rho_gcc: float) -> float:
    """``a/R*`` from the period and the mean stellar density (circular orbit)."""
    rho_si = rho_gcc * 1000.0  # g/cm^3 -> kg/m^3
    p = period_days * DAY_S
    return float((G_SI * rho_si * p ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0))


def a_over_rs_from_mr(period_days: float, mass: float, radius: float) -> float:
    p = period_days * DAY_S
    a = (G_SI * mass * MSUN_KG * p ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    return float(a / (radius * RSUN_M))


def rho_circ(period_days: float, a_rs: float) -> float:
    """The density implied by the transit, assuming a circular orbit, in g/cm^3.

    ``rho_circ = 3*pi/(G P^2) * (a/R*)^3``.  Comparing this with the star's
    own density is the duration test; the *ratio* is the quantity to report.
    """
    p = period_days * DAY_S
    rho_si = 3.0 * np.pi * a_rs ** 3 / (G_SI * p ** 2)
    return float(rho_si / 1000.0)


def duration_t14(period_days: float, a_rs: float, rp_rs: float = 0.0,
                 b: float = 0.0, ecc: float = 0.0, omega_deg: float = 90.0) -> float:
    """Total (first-to-fourth contact) transit duration, in days."""
    if a_rs <= 1:
        return float("nan")
    omega = np.radians(omega_deg)
    ecc_factor = np.sqrt(max(1 - ecc ** 2, 1e-9)) / (1 + ecc * np.sin(omega))
    arg_sq = (1 + rp_rs) ** 2 - b ** 2
    if arg_sq <= 0:
        return 0.0
    sini = np.sqrt(max(1 - (b / a_rs) ** 2, 1e-9))
    arg = np.clip(np.sqrt(arg_sq) / (a_rs * sini), 0, 1)
    return float((period_days / np.pi) * np.arcsin(arg) * ecc_factor)


def duration_range(period_days: float, rho_gcc: float, rp_rs: float = 0.0) -> tuple[float, float]:
    """``(T14_min, T14_max)`` in days allowed by this density, from b=1 to b=0.

    The lower bound is a grazing transit, which for a finite planet is not
    zero; the upper bound is a central crossing.
    """
    a = a_over_rs(period_days, rho_gcc)
    t_max = duration_t14(period_days, a, rp_rs, b=0.0)
    t_min = duration_t14(period_days, a, rp_rs, b=1.0)
    return float(t_min), float(t_max)


def implied_b(period_days: float, rho_gcc: float, t14_days: float,
              rp_rs: float = 0.0) -> float | None:
    """The impact parameter that reproduces ``t14_days`` at this density.

    Returns ``None`` when no ``b`` in [0, 1+k] can do it, which is the only
    case where the duration is actually in tension with the star.  Asking this
    first matters: a transit that is *shorter* than a central crossing needs no
    eccentricity at all, only a larger ``b``, and a test that jumps straight to
    :func:`reconcile_density` reports a spurious e ~ 0.3 for perfectly ordinary
    grazing geometry (DS Tuc A b does exactly this).

    Inverting ``T14 = (P/pi) asin(sqrt((1+k)^2 - b^2) / (a sin i))`` with
    ``sin i = sqrt(1 - (b/a)^2)`` gives
    ``b^2 = [(1+k)^2 - s^2 a^2] / (1 - s^2)``, ``s = sin(pi T14 / P)``.
    """
    if not (period_days > 0 and rho_gcc and rho_gcc > 0 and t14_days > 0):
        return None
    a = a_over_rs(period_days, rho_gcc)
    s = np.sin(np.pi * t14_days / period_days)
    if s >= 1:
        return None
    b_sq = ((1 + rp_rs) ** 2 - s ** 2 * a ** 2) / (1 - s ** 2)
    if b_sq < 0:
        return None          # even a central crossing is too short: too long a transit
    b = float(np.sqrt(b_sq))
    return b if b <= 1 + rp_rs else None   # too short: would have to miss the star


def reconcile_density(rho_star: float, rho_transit: float) -> dict:
    """What it would take to reconcile a transit-implied density with the star's.

    Reported instead of a veto.  Young-star radii are poor enough that a 20%
    radius error or a mild eccentricity is entirely ordinary, and a duration
    test that vetoes on the ratio alone throws away real planets.

    * ``ecc_required`` is the *minimum* eccentricity that can produce the
      observed ratio, achieved at the optimal argument of periastron.  Writing
      ``g = (rho_transit/rho_star)^(1/3)``, that is ``|g^2-1| / (g^2+1)``.
    * ``radius_error_required`` is the fractional change in R* (at fixed mass)
      that removes the discrepancy: ``1/g - 1``.  Positive means the star
      would have to be *larger* than catalogued.
    """
    if not (rho_star and rho_transit and rho_star > 0 and rho_transit > 0):
        return {"ratio": None, "ecc_required": None, "radius_error_required": None}
    ratio = rho_transit / rho_star
    g = ratio ** (1.0 / 3.0)
    ecc = abs(g ** 2 - 1.0) / (g ** 2 + 1.0)
    return {
        "ratio": float(ratio),
        "log10_ratio": float(np.log10(ratio)),
        "ecc_required": float(ecc),
        "radius_error_required": float(1.0 / g - 1.0),
        "note": ("transit is shorter than the star's density allows"
                 if ratio > 1 else
                 "transit is longer than the star's density allows"),
    }


def planet_radius_earth(rp_rs: float, r_star_rsun: float) -> float:
    """R_p in Earth radii.  1 R_sun = 109.076 R_earth."""
    return float(rp_rs * r_star_rsun * 109.076)


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _int(x):
    """Exactly, when ``x`` is already an integer or its string: a Gaia source_id
    does not survive float(), which rounds 78% of them by up to 512."""
    try:
        return int(x)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None
