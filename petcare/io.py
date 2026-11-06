"""Light-curve loading.

One entry point, :func:`load_lightcurve`, which sniffs the format rather than
assuming one.  Supported:

``.npz``
    The tessquicklook per-sector product.  This is the primary format
    for this project: one file per star per sector, already normalised to a
    median of 1, carrying the aperture mask, the dilution, the rotation
    measurement and the per-sector noise statistics.
``.csv`` / ``.txt`` / ``.dat``
    Either the tessquicklook CSV (``time,flux,flux_med,flux_raw,flux_flat,...``)
    or a plain two/three-column ``time,flux[,flux_err]`` table, with or without
    a header.
``.fits``
    SPOC (``PDCSAP_FLUX``/``SAP_FLUX``) or QLP (``KSPSAP_FLUX``/``DET_FLUX``).

Several files load and stitch into one :class:`LightCurve`; each sector keeps
its own entry in ``.sectors`` so per-sector tests (``pc_instrument``) can still
take it apart.

Flux column policy
------------------
The default is the *systematics-corrected but not flattened* flux (``fcor``
in the npz, ``flux`` in the tessquicklook CSV, ``PDCSAP_FLUX`` in SPOC).
Stellar variability is left in on purpose: every model-fitting test in this
package fits a GP and the transit simultaneously, because flattening first
distorts real signal on rapidly-rotating young stars.  ``flux_column="fflat"``
is available for quick looks but must not be fed to ``pc_fit``.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------

#: Roots searched by :func:`find_lightcurves` when none are given explicitly.
#: Set ``PETCARE_LC_ROOTS`` to a colon-separated list of directories.  The
#: package itself hardcodes no paths -- the local driver supplies them.
ENV_LC_ROOTS = "PETCARE_LC_ROOTS"

_SECTOR_RE = re.compile(r"[_-]s(\d{2,4})", re.IGNORECASE)


class LightCurveError(Exception):
    """Raised when a file cannot be read as a light curve."""


# --------------------------------------------------------------------------


#: Shortest rotation period searched, and the bin the search runs on.
LS_P_MIN_DAYS = 0.1
LS_BIN_DAYS = 30.0 / 1440.0
#: A segment must span this many cycles, because the periodogram of a shorter
#: stretch rails at its upper limit whatever the star is doing.
LS_MIN_CYCLES = 3.0
LS_MIN_BINS = 100
#: The TESS orbit, which systematics fold onto as convincingly as a star does.
TESS_ORBIT_DAYS = 13.7
TESS_ORBIT_TOL = 0.06


def measure_prot_ls(t, f, sector_id) -> dict:
    """Rotation period from a per-sector Lomb-Scargle, or None if none is clear.

    Each sector is searched separately and the median taken, so one sector
    landing on a harmonic does not decide the answer.  This is a quick estimate
    for the GP kernel and the baseline window, and it can land on P/2 of a
    double-dip star.  Give ``--prot`` when you know better.
    """
    from astropy.timeseries import LombScargle

    empty = {"prot_days": None, "source": None, "n_sectors": 0, "acf_days": [],
             "ls_days": [], "sector_scatter_frac": None}
    found = []
    for sec in np.unique(sector_id):
        m = (sector_id == sec) & np.isfinite(t) & np.isfinite(f)
        if m.sum() < LS_MIN_BINS:
            continue
        ts, fs = t[m], f[m]
        idx = np.floor((ts - ts.min()) / LS_BIN_DAYS).astype(int)
        keep = np.unique(idx)
        if keep.size < LS_MIN_BINS:
            continue
        # medians, so a flare does not become power at its own recurrence
        tb = np.array([ts[idx == i].mean() for i in keep])
        fb = np.array([np.median(fs[idx == i]) for i in keep])
        p_max = float(np.ptp(tb)) / LS_MIN_CYCLES
        if p_max <= LS_P_MIN_DAYS * 2:
            continue
        freq = np.linspace(1.0 / p_max, 1.0 / LS_P_MIN_DAYS, 20000)
        power = LombScargle(tb, fb).power(freq)
        best = int(np.argmax(power))
        if best in (0, freq.size - 1):
            continue                     # railed: an upper limit, not a period
        period = float(1.0 / freq[best])
        if abs(period / TESS_ORBIT_DAYS - 1.0) < TESS_ORBIT_TOL:
            continue
        found.append(period)
    if not found:
        return empty
    prot = float(np.median(found))
    return {"prot_days": prot, "source": "ls_measured", "n_sectors": len(found),
            "acf_days": [], "ls_days": found,
            "sector_scatter_frac": (float(np.std(found) / prot)
                                    if len(found) > 1 else None)}


@dataclass
class SectorInfo:
    """Per-sector bookkeeping, kept so tests can take a stitched curve apart."""

    sector: int | None
    path: str
    n_points: int
    t_start: float
    t_stop: float
    cadence_s: float | None = None
    camera: int | None = None
    ccd: int | None = None
    dilution: float | None = None
    dilution_source: str | None = None
    aperture_id: str | None = None
    aperture_mask: np.ndarray | None = None
    cdpp_1hr_ppm: float | None = None
    prot_ls_days: float | None = None
    prot_acf_days: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class LightCurve:
    """A stitched, normalised light curve.

    ``t`` is BTJD (BJD - 2457000) for TESS products.  ``f`` is normalised to a
    median of 1 per sector before stitching, so a depth read off it is a
    fractional depth directly.
    """

    t: np.ndarray
    f: np.ndarray
    err: np.ndarray
    cadence_s: np.ndarray
    sector_id: np.ndarray            # per-point sector, for per-sector tests
    sectors: list = field(default_factory=list)   # list[SectorInfo]
    flux_column: str = "fcor"
    target: dict = field(default_factory=dict)
    paths: list = field(default_factory=list)
    prot_user: float | None = None   # --prot, which outranks any measurement
    _prot_measured: dict | None = field(default=None, repr=False)

    # -- basics --------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def baseline_days(self) -> float:
        return float(np.ptp(self.t)) if self.t.size > 1 else 0.0

    @property
    def n_sectors(self) -> int:
        return len({s.sector for s in self.sectors if s.sector is not None})

    @property
    def median_cadence_s(self) -> float:
        good = np.isfinite(self.cadence_s)
        if not good.any():
            # fall back on the time sampling itself
            d = np.diff(np.sort(self.t))
            d = d[np.isfinite(d) & (d > 0)]
            return float(np.median(d) * 86400.0) if d.size else float("nan")
        return float(np.median(self.cadence_s[good]))

    @property
    def exp_time_days(self) -> float:
        return self.median_cadence_s / 86400.0

    def dilution(self) -> tuple[float | None, str | None]:
        """Point-weighted mean target-flux fraction across sectors.

        Higher is better: 1.0 means no contaminating flux in the aperture.  The
        flux is already corrected for it, so never divide a depth by this --
        see :class:`petcare.result.Dilution`.
        """
        vals, wts, srcs = [], [], set()
        for s in self.sectors:
            if s.dilution is not None and np.isfinite(s.dilution):
                vals.append(s.dilution)
                wts.append(max(s.n_points, 1))
                if s.dilution_source:
                    srcs.add(s.dilution_source)
        if not vals:
            return None, None
        factor = float(np.average(vals, weights=wts))
        return factor, "+".join(sorted(srcs)) if srcs else None

    def rotation_period(self) -> dict:
        """Rotation period from the per-sector measurements in the extraction.

        Prefers the **ACF** over the Lomb-Scargle value.  On this sample the
        two disagree often and in a specific way: LS lands on a harmonic
        (HIP 67522's sector 11 gives 0.709 d against a true 1.418 d) and its
        argmax can rail at the search's upper period limit.  The ACF value was
        stable to better than 1% across five sectors on the same star.

        Returns the median across sectors, the scatter between them, and the
        LS comparison, so a caller can see the disagreement rather than
        inherit one number.  Everything may be None; no caller may assume a
        rotation period exists.

        A period the user supplied wins outright.  Products that carry no
        rotation period (SPOC and QLP FITS, plain text) fall back to a
        Lomb-Scargle measurement on this light curve; see
        :func:`measure_prot_ls`.
        """
        if self.prot_user:
            return {"prot_days": float(self.prot_user), "source": "user",
                    "n_sectors": 0, "acf_days": [], "ls_days": [],
                    "sector_scatter_frac": None}
        acf = [s.prot_acf_days for s in self.sectors
               if s.prot_acf_days and np.isfinite(s.prot_acf_days)]
        ls = [s.prot_ls_days for s in self.sectors
              if s.prot_ls_days and np.isfinite(s.prot_ls_days)]
        # An ACF period at or below a couple of cadences is the autocorrelation
        # of the noise, not the star.
        floor = max(4 * self.median_cadence_s / 86400.0, 0.05)
        acf = [v for v in acf if v > floor]

        if not acf and not ls:
            if self._prot_measured is None:
                self._prot_measured = measure_prot_ls(self.t, self.f, self.sector_id)
            return dict(self._prot_measured)

        prot = float(np.median(acf)) if acf else (float(np.median(ls)) if ls else None)
        source = "acf" if acf else ("ls" if ls else None)
        out = {
            "prot_days": prot, "source": source,
            "n_sectors": len(acf) if acf else len(ls),
            "acf_days": [float(v) for v in acf],
            "ls_days": [float(v) for v in ls],
            "sector_scatter_frac": (float(np.std(acf) / np.median(acf))
                                    if len(acf) > 1 and np.median(acf) > 0 else None),
        }
        if prot and ls:
            ratio = float(np.median(ls)) / prot
            out["ls_over_acf"] = ratio
            out["ls_is_harmonic"] = bool(
                any(abs(ratio - r) < 0.08 for r in (0.5, 2.0, 1 / 3, 3.0)))
        return out

    # -- selection -----------------------------------------------------------

    def sector_mask(self, sector: int) -> np.ndarray:
        return self.sector_id == sector

    def select(self, mask: np.ndarray) -> "LightCurve":
        """A copy restricted to ``mask``, keeping the per-sector records."""
        mask = np.asarray(mask, dtype=bool)
        kept = {int(s) for s in np.unique(self.sector_id[mask]) if s >= 0}
        return LightCurve(
            t=self.t[mask], f=self.f[mask], err=self.err[mask],
            cadence_s=self.cadence_s[mask], sector_id=self.sector_id[mask],
            sectors=[s for s in self.sectors if s.sector in kept],
            flux_column=self.flux_column, target=dict(self.target),
            prot_user=self.prot_user,
            paths=list(self.paths),
        )

    def describe(self) -> str:
        secs = ",".join(str(s.sector) for s in self.sectors if s.sector is not None)
        return (f"{len(self)} points, {self.n_sectors} sectors [{secs}], "
                f"{self.baseline_days:.1f} d baseline, "
                f"{self.median_cadence_s:.0f} s cadence, column={self.flux_column}")


# --------------------------------------------------------------------------
# sniffing


def sniff_format(path) -> str:
    """Return ``'npz'``, ``'fits'``, ``'tql_csv'`` or ``'plain_csv'``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix in (".fits", ".fit", ".fz") or path.name.endswith(".fits.gz"):
        return "fits"
    if suffix in (".csv", ".txt", ".dat", ".tbl"):
        try:
            with open(path, "r", errors="replace") as fh:
                head = fh.readline().strip().lower()
        except OSError as exc:
            raise LightCurveError(f"cannot read {path}: {exc}") from exc
        if "flux_err_photon" in head or "flux_flat" in head:
            return "tql_csv"
        return "plain_csv"
    raise LightCurveError(f"unrecognised light-curve format: {path}")


# --------------------------------------------------------------------------
# per-format readers.  Each returns (dict of arrays, SectorInfo).


def _low_runs(t, f, gm, window_d=0.5, min_run=3):
    """Masked cadences to restore: below the running median of the kept flux,
    in runs of at least ``min_run`` consecutive masked-and-low cadences.

    A transit bottom is a contiguous run below the baseline; a single low
    cadence is a cosmic ray or a pointing glitch and stays out.
    """
    out = np.zeros(t.size, dtype=bool)
    bad = ~gm & np.isfinite(t) & np.isfinite(f)
    if not bad.any() or gm.sum() < 10:
        return out
    tg, fg = t[gm], f[gm]
    o = np.argsort(tg)
    tg, fg = tg[o], fg[o]
    idx = np.flatnonzero(bad)
    lo = np.searchsorted(tg, t[idx] - window_d / 2)
    hi = np.searchsorted(tg, t[idx] + window_d / 2)
    low = np.zeros(idx.size, dtype=bool)
    for k, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 5:
            low[k] = f[idx[k]] < np.median(fg[a:b])
    cand = np.zeros(t.size, dtype=bool)
    cand[idx[low]] = True
    # runs in cadence order, so a gap does not join two runs
    o = np.argsort(t)
    c = cand[o]
    edges = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_run:
            out[o[a:b]] = True
    return out


def _read_npz(path, flux_column: str):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    column = flux_column
    if column not in keys:
        # 'flux'/'fcor' are the same thing under two names depending on whether
        # the caller is thinking in CSV columns or npz keys
        alias = {"flux": "fcor", "flux_flat": "fflat", "flux_raw": "f",
                 "flux_med": "fcormed"}.get(column)
        if alias in keys:
            column = alias
        else:
            raise LightCurveError(
                f"{path}: no flux column {flux_column!r}; have {sorted(keys)}")

    t = np.asarray(d["t"], dtype=float)
    f = np.asarray(d[column], dtype=float)

    # Per-point photon error, inflated to the empirical scatter when that is
    # larger.  err_empirical is a scalar point-to-point scatter for the sector
    # and is routinely 2-4x the photon error on active stars; using the photon
    # error alone there produces the "40 sigma detection of nothing" failure
    # mode.  Tests that need a red-noise-aware error do their own thing.
    err = np.asarray(d["err_photon"], dtype=float) if "err_photon" in keys else None
    if err is None or not np.isfinite(err).any():
        err = np.full_like(f, np.nan)
    if "err_empirical" in keys:
        emp = float(np.atleast_1d(d["err_empirical"])[0])
        if np.isfinite(emp) and emp > 0:
            err = np.maximum(err, emp)

    cad = (np.asarray(d["cadence_s"], dtype=float) if "cadence_s" in keys
           else np.full_like(t, np.nan))

    good = np.isfinite(t) & np.isfinite(f)
    if "fit_good_mask" in keys:
        gm = np.asarray(d["fit_good_mask"], dtype=bool)
        if gm.shape == t.shape:
            # fit_good_mask is the systematics fit's TWO-sided sigma clip, not a
            # quality mask: it drops flares and transit bottoms alike.  Only the
            # high side is thrown away; low runs come back (see _low_runs).
            if os.environ.get("PETCARE_ONESIDED_MASK", "1") != "0":
                gm = gm | _low_runs(t, f, gm & good)
            good &= gm

    def _scalar(key):
        if key not in keys:
            return None
        v = np.atleast_1d(d[key])
        if v.size != 1:
            return None
        v = v[0]
        if isinstance(v, (np.floating, float)):
            v = float(v)
            return v if np.isfinite(v) else None
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        return str(v)

    meta = {k: _scalar(k) for k in (
        "tessquicklook_version", "tessquicklook_git_sha", "cadence_path",
        "gaia_dr3_source_id", "primary_region", "settings_json",
        "scatter_ptp_scatter_ppm", "scatter_cdpp_0p5hr_ppm",
        "scatter_cdpp_2hr_ppm", "scatter_cdpp_13hr_ppm",
        "rot_prot_ls_amp_ppm", "rot_prot_ls_power", "rot_prot_acf_strength",
        "rot_prot_methods_disagree", "var_Q_metric", "var_M_metric",
        "var_is_dipper", "var_is_burster", "timecorr_fallback",
        "shared_design_key",
    ) if k in keys}
    # any caveat_* flag the extraction chose to record
    meta.update({k: _scalar(k) for k in keys if k.startswith("caveat_")})

    apmask = None
    if "aperture_mask" in keys:
        apmask = np.asarray(d["aperture_mask"], dtype=float)
        # 20 s SPOC products record aperture_id='spoc_sap' and store an empty
        # array: the aperture belongs to the SPOC pipeline and is not carried
        # here.  Absent, not empty -- a zero-size mask must not reach a caller
        # that expects a 2-D image.
        if apmask.ndim != 2 or apmask.size == 0:
            apmask = None

    info = SectorInfo(
        sector=_scalar("sector"), path=str(path), n_points=int(good.sum()),
        t_start=float(np.nanmin(t[good])) if good.any() else float("nan"),
        t_stop=float(np.nanmax(t[good])) if good.any() else float("nan"),
        cadence_s=float(np.nanmedian(cad[good])) if good.any() and np.isfinite(cad[good]).any() else None,
        camera=_scalar("camera"), ccd=_scalar("ccd"),
        dilution=_scalar("dilution"), dilution_source="tessquicklook",
        aperture_id=_scalar("aperture_id"), aperture_mask=apmask,
        cdpp_1hr_ppm=_scalar("scatter_cdpp_1hr_ppm"),
        prot_ls_days=_scalar("rot_prot_ls_days"),
        prot_acf_days=_scalar("rot_prot_acf_days"),
        meta=meta,
    )
    target = {}
    if "tic_id" in keys:
        target["tic"] = _scalar("tic_id")
    if "gaia_dr3_source_id" in keys:
        target["gaia_dr3"] = _scalar("gaia_dr3_source_id")

    return dict(t=t[good], f=f[good], err=err[good], cadence_s=cad[good]), info, target


def _read_tql_csv(path, flux_column: str):
    import csv as _csv

    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    header = [h.strip().lower() for h in rows[0]]
    col = {"fcor": "flux", "fflat": "flux_flat", "f": "flux_raw",
           "fcormed": "flux_med"}.get(flux_column, flux_column)
    if col not in header:
        raise LightCurveError(f"{path}: no column {col!r}; have {header}")

    def grab(name):
        if name not in header:
            return None
        j = header.index(name)
        return np.array([float(r[j]) if r[j] not in ("", "nan") else np.nan
                         for r in rows[1:]])

    t = grab("time")
    f = grab(col)
    err = grab("flux_err_photon")
    emp = grab("flux_err_empirical")
    if err is None:
        err = np.full_like(t, np.nan)
    if emp is not None and np.isfinite(emp).any():
        err = np.fmax(err, emp)
    cad = grab("cadence_s")
    if cad is None:
        cad = np.full_like(t, np.nan)

    # A stitched extraction can carry its own per-point sector column, and
    # when it does that is far better than the one number a filename gives:
    # pc_instrument's sector-to-sector depth and camera checks, and the GP's
    # per-sector segmentation, all work from it.  Without it every cadence
    # reads as sector -1 and those tests go dark on stitched input.
    sec = grab("sector")

    good = np.isfinite(t) & np.isfinite(f)
    out = dict(t=t[good], f=f[good], err=err[good], cadence_s=cad[good])
    if sec is not None and np.isfinite(sec[good]).any():
        out["sector_id"] = sec[good].astype(int)
        present = sorted({int(v) for v in sec[good] if np.isfinite(v)})
    else:
        present = []
    info = SectorInfo(
        sector=(present[0] if len(present) == 1 else _sector_from_name(path)),
        path=str(path), n_points=int(good.sum()),
        t_start=float(np.nanmin(t[good])) if good.any() else float("nan"),
        t_stop=float(np.nanmax(t[good])) if good.any() else float("nan"),
        cadence_s=float(np.nanmedian(cad[good])) if good.any() and np.isfinite(cad[good]).any() else None,
        meta={"sectors_present": present} if present else {},
    )
    return out, info, {}


def _read_plain_csv(path, flux_column: str):
    """``time,flux[,flux_err]``, whitespace or comma separated, header optional."""
    t = f = err = None
    # genfromtxt's default splits on whitespace only: a comma file then has one
    # "column", the named read fails, and columns get assigned by position.
    with open(path) as fh:
        first = next((ln for ln in fh if ln.strip() and not ln.startswith("#")), "")
    delim = "," if "," in first else None
    try:
        arr = np.genfromtxt(path, delimiter=delim, names=True, dtype=float,
                            comments="#", encoding=None)
        names = list(arr.dtype.names or [])
    except Exception:  # noqa: BLE001 - fall through to the headerless read
        arr, names = None, []

    if arr is not None and len(names) >= 2:
        low = [n.lower() for n in names]

        def pick(*cands):
            for c in cands:
                if c in low:
                    return names[low.index(c)]
            return None

        tcol = pick("time", "t", "bjd", "btjd", "jd")
        fcol = pick(flux_column.lower(), "flux", "f", "fcor", "norm_flux", "sap_flux")
        ecol = pick("flux_err", "err", "error", "e_flux", "sigma")
        if tcol and fcol:
            t = np.asarray(arr[tcol], float)
            f = np.asarray(arr[fcol], float)
            err = np.asarray(arr[ecol], float) if ecol else np.full_like(t, np.nan)

    if t is None:
        raw = np.genfromtxt(path, delimiter=None, comments="#")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raw = np.genfromtxt(path, delimiter=",", comments="#")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise LightCurveError(f"{path}: need at least two columns")
        t, f = raw[:, 0].astype(float), raw[:, 1].astype(float)
        err = raw[:, 2].astype(float) if raw.shape[1] > 2 else np.full_like(t, np.nan)

    # A time column in full BJD is common in hand-made files; convert so every
    # ephemeris in this package can stay in one system.
    if np.nanmedian(t) > 2_400_000:
        t = t - 2_457_000.0

    # Normalise to a median of 1 if the file is in raw counts.
    med = np.nanmedian(f)
    if np.isfinite(med) and med > 0 and not (0.5 < med < 2.0):
        f = f / med
        err = err / med

    good = np.isfinite(t) & np.isfinite(f)
    info = SectorInfo(
        sector=_sector_from_name(path), path=str(path), n_points=int(good.sum()),
        t_start=float(np.nanmin(t[good])) if good.any() else float("nan"),
        t_stop=float(np.nanmax(t[good])) if good.any() else float("nan"),
    )
    return dict(t=t[good], f=f[good], err=err[good],
                cadence_s=np.full(int(good.sum()), np.nan)), info, {}


def _read_fits(path, flux_column: str):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdr0 = hdul[0].header
        hdu = None
        for h in hdul[1:]:
            if getattr(h, "data", None) is not None and h.data.dtype.names:
                if "TIME" in h.data.dtype.names:
                    hdu = h
                    break
        if hdu is None:
            raise LightCurveError(f"{path}: no table extension with a TIME column")

        names = list(hdu.data.dtype.names)
        hdr = hdu.header

        preferred = {"fcor": ["PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX", "SAP_FLUX"],
                     "f": ["SAP_FLUX", "RAW_FLUX"],
                     "fflat": ["KSPSAP_FLUX", "DET_FLUX", "PDCSAP_FLUX"]}
        cands = preferred.get(flux_column, [flux_column.upper()]) + [
            "PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX", "SAP_FLUX"]
        fcol = next((c for c in cands if c in names), None)
        if fcol is None:
            raise LightCurveError(f"{path}: no usable flux column in {names}")

        t = np.asarray(hdu.data["TIME"], float)
        f = np.asarray(hdu.data[fcol], float)
        ecol = fcol + "_ERR"
        err = (np.asarray(hdu.data[ecol], float) if ecol in names
               else np.full_like(t, np.nan))

        good = np.isfinite(t) & np.isfinite(f)
        if "QUALITY" in names:
            good &= np.asarray(hdu.data["QUALITY"], int) == 0

        med = np.nanmedian(f[good]) if good.any() else np.nan
        if np.isfinite(med) and med != 0:
            f = f / med
            err = err / med

        exptime = hdr.get("TIMEDEL")
        cad_s = float(exptime) * 86400.0 if exptime else (
            float(hdr0.get("EXPOSURE", np.nan)) if hdr0.get("EXPOSURE") else np.nan)

        sector = hdr0.get("SECTOR", hdr.get("SECTOR")) or _sector_from_name(path)
        crowdsap = hdr.get("CROWDSAP")

        info = SectorInfo(
            sector=int(sector) if sector is not None else None,
            path=str(path), n_points=int(good.sum()),
            t_start=float(np.nanmin(t[good])) if good.any() else float("nan"),
            t_stop=float(np.nanmax(t[good])) if good.any() else float("nan"),
            cadence_s=cad_s if np.isfinite(cad_s) else None,
            camera=hdr0.get("CAMERA"), ccd=hdr0.get("CCD"),
            dilution=float(crowdsap) if crowdsap is not None else None,
            dilution_source="CROWDSAP" if crowdsap is not None else None,
            meta={"flux_column_fits": fcol, "origin": hdr0.get("ORIGIN"),
                  "telescop": hdr0.get("TELESCOP")},
        )
        target = {}
        if hdr0.get("TICID"):
            target["tic"] = int(hdr0["TICID"])
        if hdr0.get("OBJECT"):
            target["name"] = str(hdr0["OBJECT"])

    return dict(t=t[good], f=f[good], err=err[good],
                cadence_s=np.full(int(good.sum()), cad_s)), info, target


_READERS = {"npz": _read_npz, "tql_csv": _read_tql_csv,
            "plain_csv": _read_plain_csv, "fits": _read_fits}


def _sector_from_name(path) -> int | None:
    m = _SECTOR_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# the public loader


def load_lightcurve(paths, flux_column: str = "fcor",
                    sectors: Sequence[int] | None = None,
                    drop_sectors: Sequence[int] | None = None) -> LightCurve:
    """Load one or more files and stitch them into a single light curve.

    Parameters
    ----------
    paths
        A path, a glob, or an iterable of either.
    flux_column
        ``'fcor'`` (default; systematics-corrected, variability retained),
        ``'fflat'`` (flattened -- quick looks only), ``'f'`` (raw).
    sectors, drop_sectors
        Keep / remove specific sectors after loading.

    Duplicate sectors are dropped, keeping the file with the most points.
    ``lightkurve`` and cached-cutout directories both hand back the same sector
    more than once when several products exist; double-counting it halves every
    apparent uncertainty downstream.
    """
    files = _expand(paths)
    if not files:
        raise LightCurveError(f"no light-curve files matched {paths!r}")

    loaded = []
    errors = []
    for p in files:
        try:
            fmt = sniff_format(p)
            chunk, info, target = _READERS[fmt](p, flux_column)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a stitch
            errors.append(f"{Path(p).name}: {type(exc).__name__}: {exc}")
            continue
        if chunk["t"].size == 0:
            errors.append(f"{Path(p).name}: no finite points")
            continue
        loaded.append((chunk, info, target))

    if not loaded:
        raise LightCurveError(
            "no readable light curve among "
            f"{len(files)} file(s): " + "; ".join(errors[:5]))

    # de-duplicate sectors, keeping the longest
    best: dict[object, tuple] = {}
    for chunk, info, target in loaded:
        key = info.sector if info.sector is not None else info.path
        if key not in best or chunk["t"].size > best[key][0]["t"].size:
            best[key] = (chunk, info, target)
    loaded = list(best.values())

    if sectors is not None:
        want = {int(s) for s in sectors}
        loaded = [x for x in loaded if x[1].sector in want]
    if drop_sectors:
        drop = {int(s) for s in drop_sectors}
        loaded = [x for x in loaded if x[1].sector not in drop]
    if not loaded:
        raise LightCurveError("sector selection removed every file")

    loaded.sort(key=lambda x: (x[1].t_start if np.isfinite(x[1].t_start) else np.inf))

    t = np.concatenate([c["t"] for c, _, _ in loaded])
    f = np.concatenate([c["f"] for c, _, _ in loaded])
    err = np.concatenate([c["err"] for c, _, _ in loaded])
    cad = np.concatenate([c["cadence_s"] for c, _, _ in loaded])
    # A per-point sector column, when the file carried one, beats the single
    # number taken from the filename -- one stitched file can span ten sectors.
    sid = np.concatenate([
        (c["sector_id"] if c.get("sector_id") is not None else
         np.full(c["t"].size, i.sector if i.sector is not None else -1, dtype=int))
        for c, i, _ in loaded])

    order = np.argsort(t, kind="stable")
    target: dict = {}
    for _, _, tgt in loaded:
        target.update({k: v for k, v in tgt.items() if v is not None})

    sectors = []
    for c, i, _ in loaded:
        present = (i.meta or {}).get("sectors_present") or []
        if len(present) <= 1:
            sectors.append(i)
            continue
        # split one stitched file's record into one per sector it contains
        for num in present:
            m = c["sector_id"] == num
            if not m.any():
                continue
            sectors.append(SectorInfo(
                sector=int(num), path=i.path, n_points=int(m.sum()),
                t_start=float(np.nanmin(c["t"][m])), t_stop=float(np.nanmax(c["t"][m])),
                cadence_s=(float(np.nanmedian(c["cadence_s"][m]))
                           if np.isfinite(c["cadence_s"][m]).any() else None),
            ))

    lc = LightCurve(
        t=t[order], f=f[order], err=err[order], cadence_s=cad[order],
        sector_id=sid[order], sectors=sectors,
        flux_column=flux_column, target=target,
        paths=[i.path for _, i, _ in loaded],
    )
    if errors:
        lc.target.setdefault("load_warnings", errors)
    return lc


def _expand(paths) -> list[str]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        p = str(p)
        if any(ch in p for ch in "*?["):
            out.extend(sorted(glob.glob(p)))
        elif os.path.isdir(p):
            for pat in ("*.npz", "*.fits", "*.fits.gz", "*.csv"):
                out.extend(sorted(glob.glob(os.path.join(p, pat))))
        else:
            out.append(p)
    # preserve order, drop repeats
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


# --------------------------------------------------------------------------
# locating a star's light curves


def lc_roots(explicit: Sequence[str] | None = None) -> list[Path]:
    """Directories to search for a star's light curves.

    ``explicit`` wins; otherwise ``$PETCARE_LC_ROOTS`` (colon-separated).
    Deliberately no built-in default: the package must stay portable, and the
    local driver is where this machine's layout is written down.
    """
    if explicit:
        return [Path(p).expanduser() for p in explicit]
    env = os.environ.get(ENV_LC_ROOTS, "")
    return [Path(p).expanduser() for p in env.split(":") if p.strip()]


def find_lightcurves(tic: int, roots: Sequence[str] | None = None) -> list[str]:
    """All light-curve files for a TIC under ``roots``.

    Matches the per-star directory layout tessquicklook batch runs write
    (``lightcurves/tic0166527623/tic0166527623_s0011.npz``), flat directories of files
    whose names contain the TIC, and MAST products under their own names, which
    carry it as 16 digits (``tess...-0000000166527623-0120-s_lc.fits``), whether
    loose, in a per-product directory, or in ``mastDownload/TESS/``.
    """
    tic = int(tic)
    padded = f"tic{tic:010d}"
    mast = f"{tic:016d}"
    found: list[str] = []
    for root in lc_roots(roots):
        if not root.exists():
            continue
        for batch in sorted([root] + [d for d in root.iterdir() if d.is_dir()]):
            star_dir = batch / padded
            if star_dir.is_dir():
                found.extend(str(p) for p in sorted(star_dir.glob("*.npz")))
                found.extend(str(p) for p in sorted(star_dir.glob("*.fits*")))
                found.extend(str(p) for p in sorted(star_dir.glob("*.csv")))
        for pat in (f"*{padded}*", f"*tic{tic}[!0-9]*", f"*TIC{tic}[!0-9]*"):
            found.extend(str(p) for p in sorted(root.glob(pat))
                         if p.is_file() and p.suffix.lower() in
                         (".npz", ".csv", ".fits", ".gz"))
        for base in (root, root / "mastDownload" / "TESS"):
            found.extend(str(p) for p in sorted(base.glob(f"*{mast}*.fits")))
            found.extend(str(p) for p in sorted(base.glob(f"*{mast}*/*{mast}*.fits")))
    seen, uniq = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq
