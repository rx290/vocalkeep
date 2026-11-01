"""Writes assets/banner-dark.svg and banner-light.svg, in the same style as the rx290 profile hero.
    python assets/make_banner.py"""
import math
import random
from pathlib import Path

W, H = 1200, 392
FONT = "Inter,Segoe UI,Arial,sans-serif"
THEMES = {
    "dark": dict(bg0="#08111F", bg1="#11243A", title="#F4F9FF", text="#A9C0D2", voice="#5EEAD4", music="#75B9FF",
                 line="#344D65", chip="#173C46", dot="#5EEAD4", node="#12283A"),
    "light": dict(bg0="#EEF6FB", bg1="#DCEAF5", title="#0B1B2B", text="#3D566B", voice="#078779", music="#3E7CC2",
                  line="#A8CBD7", chip="#CDEBE6", dot="#078779", node="#FFFFFF"),
}


def bars(x0, n, cy, step, rnd, voice_only):
    out = []
    for i in range(n):
        t = i / n
        env = 0.35 + 0.65 * abs(math.sin(t * math.pi * 2.3 + 0.6)) * (0.6 + 0.4 * rnd.random())
        is_voice = rnd.random() < 0.55
        h = max(6.0, 112 * env)
        out.append((x0 + i * step, cy, h, is_voice))
    return out


def svg(c):
    rnd = random.Random(7)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="VocalKeep: keep the voice, remove the music. Offline, on your own GPU.">',
         '<defs>',
         f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{c["bg0"]}"/><stop offset="1" stop-color="{c["bg1"]}"/></linearGradient>',
         f'<radialGradient id="halo"><stop stop-color="{c["voice"]}" stop-opacity=".16"/><stop offset="1" stop-color="{c["voice"]}" stop-opacity="0"/></radialGradient>',
         f'<clipPath id="round"><rect width="{W}" height="{H}" rx="24"/></clipPath>',
         '</defs><g clip-path="url(#round)">',
         f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
         f'<circle cx="905" cy="196" r="300" fill="url(#halo)"/>']
    for gx in range(22, 560, 38):                                    # the hero's dot grid, bottom left
        for gy in (316, 354):
            p.append(f'<circle cx="{gx}" cy="{gy}" r="1.15" fill="{c["dot"]}" opacity="0.065"/>')
    # text block
    p += [f'<text x="72" y="78" fill="{c["voice"]}" font-family="{FONT}" font-size="15" font-weight="650" letter-spacing="2.1">OFFLINE / GPU / OPEN SOURCE</text>',
          f'<text x="70" y="170" fill="{c["title"]}" font-family="{FONT}" font-size="78" font-weight="800" letter-spacing="-2.2">VocalKeep</text>',
          f'<text x="72" y="214" fill="{c["text"]}" font-family="{FONT}" font-size="22" font-weight="450">Keep the voice. Remove the music.</text>',
          f'<text x="72" y="246" fill="{c["text"]}" font-family="{FONT}" font-size="17" font-weight="450">Any video, on your own machine: NVIDIA, AMD or CPU.</text>']
    x = 72
    for label in ("DEMUCS", "ROCM VERIFIED", "LOCAL WEB UI", "NO UPLOADS"):
        w = 26 + len(label) * 9.4
        p += [f'<rect x="{x}" y="292" width="{w:.0f}" height="34" rx="9" fill="{c["chip"]}"/>',
              f'<text x="{x + 11}" y="314" fill="{c["voice"]}" font-family="{FONT}" font-size="12" font-weight="700" letter-spacing="1.4">{label}</text>']
        x += w + 10
    # waveform: mixed in (left), filter node, voice out (right)
    cy = 196
    mixed = bars(640, 18, cy, 9, rnd, False)
    for bx, by, h, v in mixed:
        col = c["voice"] if v else c["music"]
        p.append(f'<rect x="{bx}" y="{by - h / 2:.1f}" width="5" height="{h:.1f}" rx="2.5" fill="{col}" opacity=".9"/>')
    p += [f'<path d="M812 {cy}H852" stroke="{c["line"]}" stroke-width="1.6" stroke-dasharray="4 4"/>',
          f'<circle cx="890" cy="{cy}" r="34" fill="{c["node"]}" stroke="{c["voice"]}" stroke-width="1.6"/>',
          f'<path d="M874 {cy - 12}h32l-12 14v12l-8 4v-16z" fill="none" stroke="{c["voice"]}" stroke-width="2" stroke-linejoin="round"/>',
          f'<path d="M928 {cy}H962" stroke="{c["line"]}" stroke-width="1.6" stroke-dasharray="4 4"/>']
    for i, (bx, by, h, v) in enumerate(mixed):                       # same bars on the right: voice kept, music gone
        ox = 972 + i * 9
        if v:
            p.append(f'<rect x="{ox}" y="{by - h / 2:.1f}" width="5" height="{h:.1f}" rx="2.5" fill="{c["voice"]}"/>')
        else:
            p.append(f'<rect x="{ox}" y="{by - 2}" width="5" height="4" rx="2" fill="{c["music"]}" opacity=".35"/>')
    p += [f'<text x="640" y="306" fill="{c["text"]}" font-family="{FONT}" font-size="12" font-weight="700" letter-spacing="1.4">VIDEO IN</text>',
          f'<text x="856" y="270" fill="{c["text"]}" font-family="{FONT}" font-size="12" font-weight="700" letter-spacing="1.4">SEPARATE</text>',
          f'<text x="972" y="306" fill="{c["voice"]}" font-family="{FONT}" font-size="12" font-weight="700" letter-spacing="1.4">VOICE KEPT</text>',
          f'<text x="972" y="326" fill="{c["text"]}" font-family="{FONT}" font-size="12" font-weight="600" letter-spacing="1">MUSIC ABOUT -55 dB</text>',
          f'<rect x="640" y="336" width="10" height="10" rx="2" fill="{c["voice"]}"/>',
          f'<text x="656" y="345" fill="{c["text"]}" font-family="{FONT}" font-size="11" letter-spacing=".6">voice</text>',
          f'<rect x="700" y="336" width="10" height="10" rx="2" fill="{c["music"]}"/>',
          f'<text x="716" y="345" fill="{c["text"]}" font-family="{FONT}" font-size="11" letter-spacing=".6">music</text>',
          '</g></svg>']
    return "\n".join(p)


if __name__ == "__main__":
    here = Path(__file__).parent
    for name, c in THEMES.items():
        (here / f"banner-{name}.svg").write_text(svg(c))
