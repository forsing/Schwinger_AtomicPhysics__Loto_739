#!/usr/bin/env python3
"""Schwinger V4 atomic physics: spektralni Greenov odziv i celi prelazi."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, STARTS = 96, 12
SHIFTS = (.1, 1., 10.)
ALPHAS = (.01, .1, 1.)

PAIRS = tuple(
    (i, j) for i in range(7) for j in range(i + 1, 7)
)

j = np.arange(8)
MODES = np.sqrt(2 / 8) * np.cos(
    np.pi * np.arange(1, 8)[:, None]
    * (j[None, :] + .5) / 8
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(numbers) != 7
                or numbers != sorted(numbers)
                or len(set(numbers)) != 7
                or not all(1 <= n <= 39 for n in numbers)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "kombinacija 7/39."
                )

            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) / 5.


def states(rows):
    return gaps(rows) @ MODES.T


def support(history):
    layout = gaps(history)
    tree = cKDTree(layout)

    nearest = tree.query(
        layout, k=2, p=1
    )[0][:, 1]

    radius = max(
        float(np.quantile(nearest, .75)),
        1e-12
    )

    return tree, radius


def supported(rows, tree, radius):
    distance = tree.query(
        gaps(rows), k=1, p=1
    )[0]

    return distance <= radius + 1e-10


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def green_operator(changes, shift):
    # Empirijski strukturni operator.
    # Nema unapred zadatih atomskih nivoa.
    h = changes.T @ changes / len(changes)

    eigenvalues, eigenvectors = np.linalg.eigh(
        (h + h.T) / 2
    )
    eigenvalues = np.maximum(eigenvalues, 0.)

    # Spektralni resolvent:
    # G = sum |n><n| / (E_n + shift).
    green = (
        eigenvectors / (eigenvalues + shift)[None, :]
    ) @ eigenvectors.T

    return h, green, eigenvalues


def features(values, reference, response):
    delta = np.atleast_2d(values) - reference

    pair = np.column_stack([
        delta[:, i] * delta[:, j]
        for i, j in PAIRS
    ])

    virtual = (
        delta[:, :, None] * response[None, None, :]
    ).reshape(len(delta), 49)

    # Skor CELOG prelaza.
    # Nema regresije prosecnih brojeva ili razmaka.
    return np.column_stack((
        delta,
        delta * delta,
        pair,
        virtual
    ))


def prepare(history, shift):
    if len(history) < WINDOW + 2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 2} prethodnih redova."
        )

    history = history[-(WINDOW + 2):]
    values = states(history)
    changes = np.diff(values, axis=0)

    h, green, levels = green_operator(
        changes, shift
    )

    gram = np.zeros((84, 84))
    rhs = np.zeros(84)

    for t in range(2, len(history)):
        response = green @ changes[t - 2]

        actual = features(
            values[t:t + 1],
            values[t - 1],
            response
        )

        other = features(
            states(replacements(history[t])),
            values[t - 1],
            response
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )
        rhs += contrast.sum(axis=0) / len(contrast)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        values[-1],
        green @ changes[-1],
        history[2:],
        h,
        green,
        levels
    )


def fit(prepared, alpha):
    gram, rhs, scale = prepared[:3]

    coefficient = np.linalg.solve(
        gram + alpha * np.eye(84),
        rhs
    ) / scale

    return (
        coefficient,
        prepared[3],
        prepared[4]
    )


def scores(rows, model):
    coefficient, reference, response = model

    result = features(
        states(rows), reference, response
    ) @ coefficient

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor atomskih kanala."
        )

    return result


def search(history, model):
    # Stvarna istorijska polazista.
    # Nema sintetickih aritmetickih nizova.
    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]

    tree, radius = support(history)
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((r, replacements(r)))
            for r in current
        ])

        raw = scores(options, model)
        raw[~supported(options, tree, radius)] = -np.inf

        values = raw.reshape(len(current), 225)
        chosen = np.argmax(values, axis=1)

        better = (
            values[np.arange(len(current)), chosen]
            > values[:, 0] + 1e-10
        )

        if not np.any(better):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations
            )

        blocks = options.reshape(
            len(current), 225, 7
        )

        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha):
    return search(
        prepared[5],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW + 2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    settings = tuple(
        (s, a) for s in SHIFTS for a in ALPHAS
    )
    validation = [[] for _ in settings]

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    for t in range(begin, split):
        index = 0

        for shift in SHIFTS:
            p = prepare(rows[:t], shift)

            for alpha in ALPHAS:
                combo, _, _ = predict(p, alpha)
                validation[index].append(
                    hits(combo, rows[t])
                )
                index += 1

        if (t - begin + 1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t - begin + 1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    chosen = int(np.argmax(means))
    shift, alpha = settings[chosen]

    print(
        f"Validacija: shift={shift:g}; "
        f"alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    model_scores, control_scores = [], []

    # Shift i alpha ostaju fiksni tokom testa.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _, _ = predict(
            prepare(rows[:t], shift),
            alpha
        )

        model_scores.append(
            hits(combo, rows[t])
        )
        control_scores.append(
            hits(rows[t - 1], rows[t])
        )

        if (t - split + 1) % 16 == 0:
            print(
                f"Test: {t - split + 1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split + 1}..{len(rows)}")

    summarize(
        "SCHWINGER V4 ATOMIC PHYSICS",
        model_scores
    )
    summarize(
        "Kontrola: prethodno izvuceno kolo",
        control_scores
    )

    p = prepare(rows, shift)
    combo, score, iterations = predict(p, alpha)

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )
    _, radius = support(p[5])

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows) + 1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        "Strukturni nivoi:",
        ",".join(f"{e:.6f}" for e in p[8])
    )
    print(
        "Empirijska podrska: "
        f"L1 poluprecnik={radius:.6f}; kvantil=.75"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Greenov odziv je strukturna analogija; "
        "skor nije verovatnoca dobitka."
    )


def self_test():
    assert np.allclose(
        MODES @ MODES.T, np.eye(7)
    )
    assert np.allclose(
        MODES.sum(axis=1), 0.
    )

    base = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39]
    ])

    g = gaps(base)
    restored = (
        g[0]
        + (states(base) - states(base[0])) @ MODES
    )

    assert np.allclose(restored, g)
    assert np.allclose(
        5 * np.cumsum(restored[:, :-1], axis=1),
        base
    )

    history = np.tile(base, (30, 1))

    for shift in SHIFTS:
        p = prepare(history, shift)
        h, green = p[6:8]

        assert np.allclose(
            (h + shift * np.eye(7)) @ green,
            np.eye(7),
            atol=1e-10
        )

        eps = 1e-5
        changes = np.diff(
            states(history[-(WINDOW + 2):]),
            axis=0
        )
        gp = green_operator(
            changes, shift + eps
        )[1]
        gm = green_operator(
            changes, shift - eps
        )[1]

        assert np.allclose(
            (gp - gm) / (2 * eps),
            -green @ green,
            atol=1e-6,
            rtol=1e-6
        )

        for alpha in ALPHAS:
            model = fit(p, alpha)
            a = predict(p, alpha)

            assert a == predict(
                prepare(history, shift), alpha
            )

            tree, radius = support(p[5])
            r = replacements(a[0])
            allowed = supported(r, tree, radius)

            assert (
                scores(r[allowed], model).max(
                    initial=-np.inf
                )
                <= a[1] + 1e-9
            )

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1], 1.), .1)
        == predict(prepare(changed[:-1], 1.), .1)
    )

    print(
        "Self-test OK: rekonstrukcija, Greenov identitet, "
        "izvod, hronologija i pretraga."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: shift=10; alpha=1; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER V4 ATOMIC PHYSICS: prosek=1.3750; pogodaka 0..7=[10, 28, 19, 6, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,2,3,4,6,12,13
Skor=1.3998196131; iteracija=23; vec postoji u istoriji=False
Strukturni nivoi: 0.694339,0.788451,0.998846,1.401454,1.635821,1.798513,1.993091
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Greenov odziv je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: shift=1; alpha=0.01; prosek=1.5833
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER V4 ATOMIC PHYSICS: prosek=1.4375; pogodaka 0..7=[14, 21, 18, 9, 2, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,20,21,24,25,26,28
Skor=3.8596894338; iteracija=6; vec postoji u istoriji=False
Strukturni nivoi: 0.663610,0.884473,1.116648,1.209249,1.279519,1.874628,2.326925
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Greenov odziv je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: shift=0.1; alpha=0.01; prosek=1.4167
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER V4 ATOMIC PHYSICS: prosek=1.4062; pogodaka 0..7=[12, 23, 21, 7, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,2,3,4,5,16,26
Skor=2.6432798208; iteracija=18; vec postoji u istoriji=False
Strukturni nivoi: 0.858255,1.021211,1.204309,1.519663,1.828091,1.879642,2.131716
Empirijska podrska: L1 poluprecnik=3.200000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Greenov odziv je strukturna analogija; skor nije verovatnoca dobitka.
"""



"""
CSV	        Sledeća kombinacija	Prosek pogodaka na 64 kola	Kontrola: prethodno izvučeno kolo
Objedinjeni	1,2,3,4,6,12,13	    1,3750	                1,2188
Loto	    1,20,21,24,25,26,28	1,4375	                1,2969
Loto Plus	1,2,3,4,5,16,26	    1,4062	                1,2969
"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
