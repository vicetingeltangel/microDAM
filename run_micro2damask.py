"""Einfacher PyCharm-Starter für micro2damask.

Die Datei in den Projekt-Hauptordner legen, also neben ``src``.
Danach nur die Werte im Abschnitt "BENUTZER-EINSTELLUNGEN" anpassen
und in PyCharm auf Run drücken.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


# ============================================================================
# Projektpfad / Import
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

# Dadurch funktioniert das Skript auch ohne vorheriges `pip install -e .`.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from micro2damask import Config, run_pipeline  # noqa: E402


# ============================================================================
# BENUTZER-EINSTELLUNGEN
# ============================================================================

# Eingangsbild
IMAGE_PATH = Path(
    r"/Users/lorenzmaier/PycharmProjects/microDAM/examples/AlSi7Mg_1 Mitte 100x.tif"
)

# YAML-Datei mit Namen und DAMASK-Materialdefinitionen der dunklen/hellen Phase.
# Liegt die Datei im Projektordner, reicht z. B.:
PHASE_CONFIG_PATH = PROJECT_DIR / "phase_config_example.yaml"

# Maßstab des Originalbildes [µm/Pixel]
UM_PER_PIXEL = 0.35

# RVE-Größe im Originalbild [Pixel]
RVE_WIDTH = 1024
RVE_HEIGHT = 1024

# Position des RVE im Originalbild.
# None = automatische Auswahl.
RVE_X = None
RVE_Y = None

# Downsampling: z. B. 4 bedeutet 4x4 Bildpixel -> 1 Voxel in x/y.
DOWNSAMPLE_FACTOR = 1

# Anzahl der Voxelschichten in z-Richtung
NZ_LAYERS = 1

# Phase, auf die die morphologische Bereinigung angewendet wird:
# "dark" oder "light"
MORPHOLOGY_TARGET_PHASE = "light"

# Ergebnisordner
OUTPUT_DIR = PROJECT_DIR / "output"

# Diagnoseplots abspeichern
SAVE_DEBUG_PLOTS = True


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def load_phase_config(path: Path) -> Dict[str, Any]:
    """Lädt Namen und DAMASK-Materialdefinitionen der beiden Bildphasen."""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Die Phasenkonfiguration muss ein YAML-Dictionary sein.")

    for phase in ("dark", "light"):
        if phase not in data or not isinstance(data[phase], dict):
            raise ValueError(
                f"In der Phasenkonfiguration fehlt der Abschnitt '{phase}'."
            )

        if not data[phase].get("name"):
            raise ValueError(f"Für '{phase}' fehlt der Phasenname 'name'.")

        if not isinstance(data[phase].get("material"), dict):
            raise ValueError(
                f"Für '{phase}' fehlt die DAMASK-Materialdefinition 'material'."
            )

    return data


def build_config(phase_cfg: Dict[str, Any]) -> Config:
    """Erzeugt die micro2damask-Konfiguration aus den Einstellungen oben."""

    return Config(
        image_path=str(IMAGE_PATH),
        um_per_pixel=UM_PER_PIXEL,

        # Phasenzuordnung:
        # phase_id 0 = dunkel, phase_id 1 = hell
        dark_phase_name=str(phase_cfg["dark"]["name"]),
        light_phase_name=str(phase_cfg["light"]["name"]),
        dark_phase_material=phase_cfg["dark"]["material"],
        light_phase_material=phase_cfg["light"]["material"],

        morphology_target_phase=MORPHOLOGY_TARGET_PHASE,

        rve_w=RVE_WIDTH,
        rve_h=RVE_HEIGHT,
        rve_x=RVE_X,
        rve_y=RVE_Y,

        downsample_factor=DOWNSAMPLE_FACTOR,
        nz_layers=NZ_LAYERS,

        output_root=str(OUTPUT_DIR),
        save_debug_plots=SAVE_DEBUG_PLOTS,
    )


# ============================================================================
# Programmstart
# ============================================================================


def main() -> None:
    print("=" * 70)
    print("micro2damask")
    print("=" * 70)

    if not SRC_DIR.is_dir():
        raise FileNotFoundError(
            f"Der src-Ordner wurde nicht gefunden:\n{SRC_DIR}\n\n"
            "Lege run_micro2damask.py in den Projekt-Hauptordner neben 'src'."
        )

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Das Eingangsbild wurde nicht gefunden:\n{IMAGE_PATH}\n\n"
            "Passe IMAGE_PATH oben im Skript an."
        )

    if not PHASE_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Die Phasenkonfiguration wurde nicht gefunden:\n"
            f"{PHASE_CONFIG_PATH}\n\n"
            "Passe PHASE_CONFIG_PATH oben im Skript an."
        )

    phase_cfg = load_phase_config(PHASE_CONFIG_PATH)
    cfg = build_config(phase_cfg)

    print(f"Bild:              {IMAGE_PATH}")
    print(f"Dunkle Phase (0):  {cfg.dark_phase_name}")
    print(f"Helle Phase  (1):  {cfg.light_phase_name}")
    print(f"RVE:               {cfg.rve_w} x {cfg.rve_h} Pixel")
    print(f"Downsampling:      {cfg.downsample_factor}")
    print(f"z-Schichten:       {cfg.nz_layers}")
    print(f"Ausgabe:           {OUTPUT_DIR}")

    print("\nPipeline wird ausgeführt ...\n")
    result = run_pipeline(cfg)

    validation = result["validation"]

    print("\n" + "=" * 70)
    print("VALIDIERUNG")
    print("=" * 70)
    print(f"Modell gültig: {validation['ok']}")

    if validation["warnings"]:
        print("\nWarnungen:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    if validation["errors"]:
        print("\nFehler:")
        for error in validation["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)
    print(f"Ergebnisordner: {result['output_dir']}")
    print(f"Materialdatei:  {result['material']['material_file']}")
    print(f"Geometriedatei: {result['damask'].get('vti_path')}")

    if validation["ok"]:
        print("\n✓ Pipeline erfolgreich abgeschlossen und validiert.")
    else:
        print("\n✗ Pipeline abgeschlossen, aber die Validierung enthält Fehler.")


if __name__ == "__main__":
    main()
