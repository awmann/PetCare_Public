"""pc_followup -- what has already been observed, and does any of it settle this?

Two archives, asked two different questions.

**ExoFOP-TESS** holds what the community has filed on the star: speckle and AO
imaging, spectroscopy, ground-based transit photometry, and any *resolved
stellar companion*.  That last one is the reason this test earns its place
next to pc_neighbours rather than duplicating it.  pc_neighbours works from
Gaia, and Gaia does not resolve pairs below roughly half an arcsecond -- so
the companion most likely to be producing a blended eclipse is precisely the
one it cannot see.  Speckle interferometry does see it.  For every companion
ExoFOP lists, this test asks pc_neighbours' question at a separation
pc_neighbours cannot reach: **how deep would that star have to eclipse?**

**The LCO science archive** is searched by position, so it finds frames
whether or not anyone filed them.  That matters because most of our
candidates are not TOIs and have no ExoFOP presence at all.  Frames landing
inside a predicted transit are called out by name: that is data already on
disk that would settle the candidate, and nobody has to apply for it.

What this test cannot do is prove a negative.  The LCO cone search sees
public frames only -- a proprietary programme's data is invisible to it -- so
an empty result means *nothing was found*, never *nothing was taken*.  It is
reported that way.

    python -m petcare.scripts.pc_followup --tic 410214986 --depth 3000
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np

from .. import cli, followup
from ..plotting import C_ALT, C_BAD, C_OK, annotate_status, figure, save
from ..result import Result

TEST = "followup"
VERSION = "0.1.1"

#: A companion needing more than this fraction eclipsed is impossible.
IMPOSSIBLE_DEPTH = 1.0

#: ...and above this it needs a near-total eclipse: possible for a pair of
#: similar stars, so constrained rather than excluded.
NEAR_TOTAL_DEPTH = 0.5

#: Dispositions somebody has already decided against.
REFUTED = {"FP", "FA"}
CONFIRMED = {"CP", "KP"}

#: Phrases in a filed observing note that name the signal as off-target.
#: Deliberately short and specific.  A note is quoted verbatim in the reason
#: whenever it drives the verdict, so a wrong match is visible rather than
#: buried -- and the matching is only ever allowed to *lower* the verdict.
OFFTARGET_PHRASES = (
    "nearby eclipsing binary", "neb on", "cleared as neb", "is an neb",
    "event on nearby", "eclipse on nearby", "off-target", "off target",
    "not on target", "on neighbor", "on neighbour",
)

#: A negation just before a phrase reverses it: "No NEB on any star" contains
#: "neb on" and used to refute the candidate.
_NEGATED = re.compile(r"\b(no|not|non|never|without|none|free of|clear of|cleared of|"
                      r"rules? out|ruled out|ruling out|exclud\w*)\b[^.;:]{0,40}$")


def _off_target_phrase(low: str):
    """The first off-target phrase in a lower-case note that is not negated."""
    for p in OFFTARGET_PHRASES:
        for m in re.finditer(re.escape(p), low):
            if not _NEGATED.search(low[:m.start()]):
                return p
    return None


#: ...and phrases that report an on-target detection.
ONTARGET_PHRASES = (
    "on target", "on-target", "detected on target", "achromatic",
    "verified on target", "transit detected",
)

#: Proposals that are not observations of anybody's target.  A calibration or
#: engineering frame can still contain the star -- the cone search matches on
#: footprint -- but it was pointed for another reason, and counting it as
#: "data taken during a transit" invents follow-up that does not exist.
NON_SCIENCE_PROPOSALS = ("calibrate", "lcoengineering", "lcoepo", "standard",
                         "auto_focus", "autofocus", "ftpepo", "sciencearchive")

#: Excluding non-science *proposals* is not enough, and this was the more
#: expensive mistake.  A cone search matches on footprint, so a perfectly
#: ordinary science programme observing something else entirely will return
#: frames that contain our star as a field object.  One target came back
#: with 100 frames over 10 nights "inside a predicted transit"; every one was
#: pointed at a pulsar or a transient elsewhere in the field, six frames a
#: night.  That is not follow-up, and reporting it as such sent somebody to
#: look for data that does not exist.
#:
#: Two conditions separate an observation *of* the target from an image that
#: merely contains it, and both are needed:
#:
#: **Cadence.**  A transit time series is tens to hundreds of frames in one
#: night.  Serendipitous coverage is one to six.  This is the stronger signal.
#:
#: **Pointing.**  The target should lie near the field centre.  Not *at* it:
#: LCO's own TFOP observations routinely sit 3-6 arcmin off, so a tight cut
#: rejects real follow-up -- an earlier 2 arcmin threshold threw out two
#: targets that ExoFOP confirms were observed. 8 arcmin keeps those and still
#: rejects the wide-field mosaics of extended objects that put our star 11-37
#: arcmin out.
#:
#: The frame's ``OBJECT`` string is deliberately *not* used.  It is free text
#: and wrong in both directions: it reads "TOI-6715" for one of our targets
#: and a bare coordinate string for another, and one target has six frames
#: correctly labelled with its own TIC that are still only a snapshot.
MIN_FRAMES_PER_NIGHT = 20
MAX_POINTING_OFFSET_ARCMIN = 8.0

#: Widen a predicted transit window by this much when testing whether an
#: archival frame falls inside it.  Absorbs the barycentric correction that
#: ``iso_to_btjd`` does not apply, plus ephemeris drift over years of baseline.
WINDOW_PAD_H = 1.0

#: Beyond this many cycles from t0, an "inside a transit" match is only as
#: good as the period.  A period error of 1e-4 d accumulates to 1.3 hours over
#: 530 epochs, which is the whole window -- so a match at a large epoch number
#: is a lead to check against the period uncertainty, not a coverage claim.
#: Archival frames predate the TESS baseline as often as not, which is why the
#: epoch numbers here run negative as well as large.
EXTRAPOLATION_EPOCHS = 200


def add_arguments(ap):
    g = ap.add_argument_group("followup")
    g.add_argument("--max-age-days", type=float, default=7.0,
                   help="re-query the archives if the cache is older")
    g.add_argument("--max-lco-frames", type=int, default=500,
                   help="stop paging LCO after this many frames")
    g.add_argument("--no-lco", action="store_true",
                   help="skip the LCO archive and use ExoFOP only")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.target.tic is not None,
                       "pc_followup needs a TIC: ExoFOP is indexed by it"):
        return r

    try:
        ex = followup.exofop_target(int(ctx.target.tic),
                                    max_age_days=float(getattr(a, "max_age_days", 7.0)))
    except Exception as exc:  # noqa: BLE001 - an archive outage is not a verdict
        return r.inconclusive(f"could not read ExoFOP for TIC {ctx.target.tic}: {exc}")

    r.set_provenance(lightcurve=None, source="ExoFOP-TESS + LCO science archive",
                     url=f"{followup.EXOFOP_TARGET}?id={ctx.target.tic}&json")
    r.set_noise(None, "not applicable: this is an archive inventory")

    imaging = ex.get("imaging") or []
    spect = ex.get("spectroscopy") or []
    tseries = ex.get("time_series") or []
    comps = ex.get("stellar_companions") or []
    tois = ex.get("tois") or []

    ra, dec = _coords(ctx, ex)

    # ---- resolved companions: pc_neighbours' question, closer in ----------
    depth = ctx.candidate.depth_ppm
    companions = []
    for c in comps:
        sep = _f(c.get("sep"))
        dmag = _f(c.get("dmag"))
        req = (followup.required_depth_on_companion(depth, dmag)
               if depth and dmag is not None else None)
        companions.append({
            "separation_arcsec": sep, "position_angle_deg": _f(c.get("pa")),
            "delta_mag": dmag, "filter": c.get("filt"),
            "required_eclipse_depth": req,
            "viable": bool(req is not None and req <= NEAR_TOTAL_DEPTH),
            "needs_near_total_eclipse": bool(
                req is not None and NEAR_TOTAL_DEPTH < req <= IMPOSSIBLE_DEPTH),
            "excluded": bool(req is not None and req > IMPOSSIBLE_DEPTH),
            "observed": c.get("odate"), "note": c.get("scnotes"),
        })
    viable = [c for c in companions if c["viable"]]
    viable.sort(key=lambda c: (c["separation_arcsec"] is None,
                               c["separation_arcsec"] or 0.0))

    # ---- filed ground-based photometry ------------------------------------
    ts_rows, adverse, supportive = [], [], []
    for t in tseries:
        note = (t.get("tsnotes") or "").strip()
        low = note.lower()
        row = {"telescope": t.get("tstel"), "filter": t.get("tsfilt"),
               "date": t.get("tsdate"), "coverage": t.get("tstcov"),
               "type": t.get("tstype"), "planet": t.get("tsplname"),
               "note": note or None}
        hit = _off_target_phrase(low)
        if hit:
            row["off_target_phrase"] = hit
            adverse.append(row)
        elif any(p in low for p in ONTARGET_PHRASES):
            supportive.append(row)
        ts_rows.append(row)

    # ---- the LCO archive, by position -------------------------------------
    lco = {"searched": False}
    if not getattr(a, "no_lco", False) and ra is not None and dec is not None:
        try:
            cap = int(getattr(a, "max_lco_frames", 500))
            frames = followup.lco_frames(
                ra, dec, max_frames=cap,
                max_age_days=float(getattr(a, "max_age_days", 7.0)))
            lco = _summarise_lco(frames, ctx, ra, dec)
            # Hitting the cap exactly means there are more frames than we
            # fetched, so the night list is a floor, not a census.
            lco["truncated"] = bool(len(frames) >= cap)
            lco["frame_cap"] = cap
        except Exception as exc:  # noqa: BLE001
            lco = {"searched": False, "error": f"{type(exc).__name__}: {exc}"}

    r.metrics.update({
        "n_imaging": len(imaging), "n_spectroscopy": len(spect),
        "n_time_series": len(tseries), "n_companions": len(comps),
        "imaging": [{"telescope": i.get("itel"), "instrument": i.get("iinst"),
                     "filter": i.get("ifilt"), "type": i.get("itype"),
                     "date": i.get("idate")} for i in imaging],
        "spectroscopy": [{"telescope": s.get("stel"), "instrument": s.get("sinst"),
                          "resolution": _f(s.get("ssres")), "date": s.get("sdate")}
                         for s in spect],
        "time_series": ts_rows,
        "companions": companions,
        "n_viable_companions": len(viable),
        "closest_viable_companion": viable[0] if viable else None,
        "tfopwg_dispositions": sorted({str(t.get("disp_tfop")) for t in tois
                                       if t.get("disp_tfop")}),
        "lco": lco,
        "depth_ppm": depth,
    })

    plot = _plot(ctx, r, companions, lco)
    if plot:
        r.add_plot(plot)

    have_anything = bool(imaging or spect or tseries or comps
                         or lco.get("n_frames"))
    r.set_self_check(
        True,
        f"{len(imaging)} imaging, {len(spect)} spectroscopy, {len(tseries)} "
        f"time-series and {len(comps)} companion row(s) on ExoFOP; "
        + (f"{lco.get('n_frames', 0)} public LCO frame(s) over "
           f"{lco.get('n_nights', 0)} night(s)" if lco.get("searched")
           else "LCO not searched"))

    return _verdict(r, ctx, imaging, spect, ts_rows, companions, viable,
                    adverse, supportive, lco, have_anything,
                    r.metrics["tfopwg_dispositions"])


# --------------------------------------------------------------------------
def _verdict(r, ctx, imaging, spect, ts_rows, companions, viable,
             adverse, supportive, lco, have_anything, dispositions=()) -> Result:
    inventory = (f"{len(imaging)} imaging, {len(spect)} spectroscopy and "
                 f"{len(ts_rows)} ground-based time-series observation(s) on "
                 f"ExoFOP")
    if lco.get("searched"):
        inventory += (f"; {'at least ' if lco.get('truncated') else ''}"
                      f"{lco.get('n_frames', 0)} public LCO frame(s) on "
                      f"{lco.get('n_nights', 0)} night(s)")
        if lco.get("n_nights_in_transit"):
            inventory += (f", covering {lco['n_nights_in_transit']} night(s) "
                          f"inside a predicted transit")
    known = [d for d in (dispositions or ()) if d in CONFIRMED]
    if known:
        inventory += (f". TFOPWG has this star as {'/'.join(known)}, so a flag "
                      f"here is a blend source somebody has already had to "
                      f"account for, not a new doubt")
    elif [d for d in (dispositions or ()) if d in REFUTED]:
        inventory += (". TFOPWG has already dispositioned this star "
                      f"{'/'.join(d for d in dispositions if d in REFUTED)}")

    if adverse:
        first = adverse[0]
        return r.flag(
            f"a filed observing note says the event is not on the target. "
            f"{first.get('telescope')} on {first.get('date')}: "
            f"\"{first.get('note')}\" (matched on \"{first['off_target_phrase']}\"). "
            f"This is somebody else's determination read out of free text -- "
            f"check the note and the underlying data before acting on it, and "
            f"be aware it may refer to a different signal on the same star. "
            f"{inventory}")

    if viable:
        c = viable[0]
        sep = c["separation_arcsec"]
        return r.flag(
            f"high-resolution imaging resolves a companion at "
            f"{sep:.2f}\" with dmag {c['delta_mag']:.1f} in {c['filter']}, "
            f"which would need only a {100 * c['required_eclipse_depth']:.1f}% "
            f"eclipse to produce this depth. Gaia does not resolve pairs this "
            f"close, so pc_neighbours cannot see this star. "
            f"{len(viable)} viable companion(s) in total. {inventory}"
            if sep is not None else
            f"imaging resolves a companion with dmag {c['delta_mag']:.1f} that "
            f"could produce this depth with a "
            f"{100 * c['required_eclipse_depth']:.1f}% eclipse. {inventory}")

    if supportive:
        s = supportive[0]
        detection = (f"a filed ground-based observation reports the transit on "
                     f"target: {s.get('telescope')} on {s.get('date')}, "
                     f"\"{s.get('note')}\"")
        refuted = [d for d in (dispositions or ()) if d in REFUTED]
        if refuted:
            # An observer's note and the working group's disposition disagree.
            # That is not a pass, and it is not this test's place to decide
            # which is right -- it is a reason for a human to read both.
            return r.flag(
                f"{detection} -- but TFOPWG has dispositioned this star "
                f"{'/'.join(refuted)}. The observer who took the data and the "
                f"working group that reviewed it disagree, so read both before "
                f"treating either as settled. {inventory}")
        return r.passed(f"{detection}. {inventory}")

    if not have_anything:
        return r.inconclusive(
            "no follow-up found. Nothing on ExoFOP and no public LCO frames "
            "cover this position. Absence of LCO data is not evidence of "
            "absence -- the cone search sees public frames only, so a "
            "proprietary programme's data would not appear here")

    if companions and not viable:
        excluded = [c for c in companions if c["excluded"]]
        if excluded:
            return r.passed(
                f"imaging resolves {len(companions)} companion(s) and none of "
                f"them can produce this depth: the faintest would need more "
                f"than a total eclipse. {inventory}")

    # Follow-up existing is not the same as follow-up bearing on the
    # candidate.  Most public LCO frames over one of our targets are survey or
    # calibration pointings that merely contain the star, and flagging on
    # those would make this test fire on nearly everything while saying
    # nothing -- the failure mode this suite already has in pc_neighbours.  So
    # the verdict is INCONCLUSIVE and the opportunity is reported instead.
    opportunity = ""
    if lco.get("n_frames_in_transit"):
        props = ", ".join(lco.get("in_transit_proposals") or []) or "unknown"
        opportunity = (
            f". {lco.get('n_nights_in_transit', 0)} night(s) of science frames "
            f"from proposal(s) {props} fall inside a predicted transit "
            f"(epoch(s) {lco.get('transit_epochs_covered')}, "
            f"{lco['n_frames_in_transit']} frames): that is data already on "
            f"disk which would settle this candidate without applying for "
            f"anything")
        if lco.get("ephemeris_extrapolated"):
            opportunity += (
                f". The furthest match is {lco['max_abs_epoch_in_transit']} "
                f"cycles from t0, where the window is narrower than the "
                f"accumulated period error, so check it against the period "
                f"uncertainty before trusting the coverage")
    return r.inconclusive(
        f"follow-up exists but none of it bears on the candidate: {inventory}. "
        f"No filed note reports an on-target detection, and no resolved "
        f"companion is ruled in or out at this depth"
        + ("" if ctx.candidate.depth_ppm else
           " (no depth was supplied, so companion eclipse depths could not be "
           "computed -- pass --depth to get them)")
        + opportunity)


def _summarise_lco(frames, ctx, ra=None, dec=None) -> dict:
    """Group frames by night, and flag any that land inside a transit."""
    nights, in_transit = defaultdict(list), []
    period = ctx.candidate.period
    t0 = ctx.candidate.t0_btjd
    dur_h = ctx.candidate.duration_h
    half = None
    if period and t0 is not None and dur_h:
        half = (float(dur_h) + 2.0 * WINDOW_PAD_H) / 24.0 / 2.0

    for f in frames:
        stamp = f.get("observation_date") or f.get("DATE_OBS")
        night = str(stamp)[:10]
        rec = {"basename": f.get("basename"), "filter": f.get("FILTER"),
               "obstype": f.get("OBSTYPE"), "date": stamp,
               "proposal": f.get("proposal_id"),
               "exposure_s": _f(f.get("exposure_time")),
               "offset_arcmin": _pointing_offset(f.get("area"), ra, dec)}
        nights[night].append(rec)

    # Second pass, and it has to be a second pass: whether a frame belongs to
    # an observation *of* the target depends on how many frames that whole
    # night holds, which is not known until every frame has been read.
    if half is not None:
        for night, recs in nights.items():
            for rec in recs:
                # Only a science exposure counts as data taken on the target.
                # An engineering or calibration frame whose footprint happens
                # to contain the star was pointed for another reason -- and so
                # was a science frame from somebody else's programme.
                if (str(rec["obstype"]).upper() != "EXPOSE"
                        or _is_non_science(rec["proposal"])
                        or not _looks_targeted(rec, nights)):
                    continue
                btjd = followup.iso_to_btjd(rec["date"])
                if btjd is None:
                    continue
                # distance from the nearest predicted mid-transit
                n = round((btjd - float(t0)) / float(period))
                mid = float(t0) + n * float(period)
                if abs(btjd - mid) <= half:
                    in_transit.append(dict(rec, epoch=int(n),
                                           hours_from_mid=round((btjd - mid) * 24.0, 3)))

    science = {k: [f for f in v if str(f.get("obstype")).upper() == "EXPOSE"
                   and not _is_non_science(f.get("proposal"))]
               for k, v in nights.items()}
    science_nights = sorted(k for k, v in science.items() if v)
    return {
        "searched": True,
        "n_frames": sum(len(v) for v in nights.values()),
        "n_nights": len(nights),
        "n_science_nights": len(science_nights),
        "science_nights": science_nights,
        "proposals": sorted({str(f.get("proposal")) for v in nights.values()
                             for f in v if f.get("proposal")}),
        "filters": sorted({str(f.get("filter")) for v in nights.values()
                           for f in v if f.get("filter")}),
        "n_frames_in_transit": len(in_transit),
        # The meaningful unit is the night.  One targeted transit observation
        # is several hundred frames, nearly all of them inside the window, so
        # a frame count describes the cadence rather than the coverage.
        "n_nights_in_transit": len({str(f["date"])[:10] for f in in_transit}),
        "max_abs_epoch_in_transit": (max(abs(f["epoch"]) for f in in_transit)
                                     if in_transit else None),
        "ephemeris_extrapolated": bool(
            in_transit and max(abs(f["epoch"]) for f in in_transit)
            > EXTRAPOLATION_EPOCHS),
        "nights_in_transit": sorted({str(f["date"])[:10] for f in in_transit}),
        "in_transit_proposals": sorted({str(f.get("proposal")) for f in in_transit
                                        if f.get("proposal")}),
        "frames_in_transit": in_transit[:40],
        "transit_epochs_covered": sorted({f["epoch"] for f in in_transit}),
        "transit_window_tested": half is not None,
        "note": ("public frames only: a proprietary programme's data does not "
                 "appear in this search. Calibration and engineering frames "
                 "are excluded from the science-night and in-transit counts: "
                 "their footprints contain the star but they were pointed for "
                 "another reason"),
    }


def _is_non_science(proposal) -> bool:
    p = str(proposal or "").lower()
    return any(tag in p for tag in NON_SCIENCE_PROPOSALS)


def _pointing_offset(area, ra, dec):
    """Arcminutes from the target to the centre of this frame's footprint.

    ``area`` is the archive's GeoJSON polygon.  Its longitudes are wrapped to
    [-180, 180], so they are un-wrapped against the target's own RA before
    averaging -- otherwise a field near 0h lands half a sky away.
    """
    if ra is None or dec is None:
        return None
    try:
        pts = area["coordinates"][0]
    except Exception:  # noqa: BLE001 - a frame without a footprint is not fatal
        return None
    ras, decs = [], []
    for p in pts:
        try:
            a, d = float(p[0]), float(p[1])
        except Exception:  # noqa: BLE001
            return None
        if a - ra > 180:
            a -= 360.0
        elif a - ra < -180:
            a += 360.0
        ras.append(a)
        decs.append(d)
    if not ras:
        return None
    cra = sum(ras) / len(ras)
    cdec = sum(decs) / len(decs)
    dra = (cra - ra) * np.cos(np.radians(dec))
    return float(np.hypot(dra, cdec - dec) * 60.0)


def _looks_targeted(rec, nights) -> bool:
    """Was this frame part of an observation *of* the target?

    Cadence and pointing together.  See the note on MIN_FRAMES_PER_NIGHT for
    why the OBJECT string is not consulted and why the pointing tolerance is
    as loose as it is.
    """
    night = str(rec["date"])[:10]
    same = [f for f in nights.get(night, [])
            if str(f.get("obstype")).upper() == "EXPOSE"
            and not _is_non_science(f.get("proposal"))]
    if len(same) < MIN_FRAMES_PER_NIGHT:
        return False
    off = rec.get("offset_arcmin")
    return off is None or off <= MAX_POINTING_OFFSET_ARCMIN


def _coords(ctx, ex):
    """The target position, from the context if it has one, else ExoFOP's."""
    ra = getattr(ctx.target, "ra", None)
    dec = getattr(ctx.target, "dec", None)
    if ra is not None and dec is not None:
        return _f(ra), _f(dec)
    c = ex.get("coordinates") or {}
    return _f(c.get("ra")), _f(c.get("dec"))


def _plot(ctx, r, companions, lco):
    """Required eclipse depth per resolved companion, and the LCO timeline."""
    have_comp = any(c["required_eclipse_depth"] is not None for c in companions)
    have_lco = bool(lco.get("searched") and lco.get("n_frames"))
    if not (have_comp or have_lco):
        return None

    n = int(have_comp) + int(have_lco)
    fig, axes = figure(nrows=n, figsize=(8.0, 3.0 * n + 0.6))
    axes = np.atleast_1d(axes)
    i = 0

    if have_comp:
        ax = axes[i]
        i += 1
        sel = [c for c in companions if c["required_eclipse_depth"] is not None]
        sel.sort(key=lambda c: c["separation_arcsec"] or 0.0)
        x = [c["separation_arcsec"] or 0.0 for c in sel]
        y = [100.0 * c["required_eclipse_depth"] for c in sel]
        colours = [C_BAD if c["viable"] else (C_ALT if c["needs_near_total_eclipse"]
                                             else C_OK) for c in sel]
        ax.scatter(x, y, c=colours, s=70, zorder=3, edgecolor="white", linewidth=0.8)
        for c, xx, yy in zip(sel, x, y):
            ax.annotate(f"dmag {c['delta_mag']:.1f}", (xx, yy),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axhline(100.0 * IMPOSSIBLE_DEPTH, color=C_BAD, ls="--", lw=1.0)
        ax.axhline(100.0 * NEAR_TOTAL_DEPTH, color=C_ALT, ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("separation of resolved companion (arcsec)")
        ax.set_ylabel("required eclipse depth (%)")
        ax.set_title("Could an imaging-resolved companion produce this depth?",
                     fontsize=10)
        ax.text(0.99, 0.04,
                "above 100%: impossible.  Gaia does not resolve pairs below ~0.5\"",
                transform=ax.transAxes, ha="right", fontsize=7, color="#666666")

    if have_lco:
        ax = axes[i]
        nights = lco.get("science_nights") or []
        intr = {f["date"][:10] for f in lco.get("frames_in_transit") or []}
        xs = np.arange(len(nights))
        cols = [C_BAD if nn in intr else C_OK for nn in nights]
        ax.bar(xs, np.ones(len(nights)), color=cols, width=0.8)
        ax.set_xticks(xs[:: max(1, len(xs) // 12)])
        ax.set_xticklabels([nights[j] for j in xs[:: max(1, len(xs) // 12)]],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel("LCO nights")
        ax.set_title(f"{lco['n_frames']} public LCO frames on "
                     f"{lco['n_nights']} night(s)"
                     + (f"; {lco['n_frames_in_transit']} inside a predicted "
                        f"transit (marked)" if lco.get("n_frames_in_transit")
                        else ""), fontsize=10)

    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "What follow-up already exists on ExoFOP and in the LCO "
              "archive, and does any of it settle this candidate?"))
    args = ap.parse_args(argv)
    ctx = cli.Context.from_args(args)
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # noqa: BLE001
        r.error(exc)
    return cli.finish(r, ctx)


if __name__ == "__main__":
    sys.exit(main())
