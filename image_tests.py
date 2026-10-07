"""Image-input checks for one AREX-2 MLX build. Images are drawn locally, so every answer is known exactly.

Usage:
  venv/bin/python image_tests.py --make                      # only draw the images into results/images/
  venv/bin/python image_tests.py builds/AREX-2-5bit          # run the questions, write results/image-<name>.json
Thinking off, temperature 0, so the same build gives the same answers each run.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
RES = ROOT / "results"
IMG = RES / "images"


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def make_images():
    IMG.mkdir(parents=True, exist_ok=True)
    # 1. invoice: small text, several lines
    im = Image.new("RGB", (900, 620), "white"); d = ImageDraw.Draw(im)
    d.text((40, 30), "HARBOR LANE SUPPLY CO.", fill="black", font=font(34))
    d.text((40, 90), "Invoice No: HL-20417", fill="black", font=font(22))
    d.text((560, 90), "Date: 14 March 2026", fill="black", font=font(22))
    d.text((40, 125), "Bill to: Northgate Bakery, 18 Mill Road", fill="black", font=font(22))
    d.line((40, 170, 860, 170), fill="black", width=2)
    rows = [("Item", "Qty", "Unit", "Amount"), ("Flour, 25 kg sack", "12", "18.50", "222.00"),
            ("Cane sugar, 10 kg", "8", "14.25", "114.00"), ("Sea salt, 5 kg", "3", "9.80", "29.40"),
            ("Dry yeast, 500 g", "20", "4.15", "83.00")]
    for i, r in enumerate(rows):
        y = 185 + i * 44
        for x, cell in zip((40, 480, 600, 740), r):
            d.text((x, y), cell, fill="black", font=font(22))
    d.line((40, 410, 860, 410), fill="black", width=2)
    d.text((560, 425), "Subtotal:   448.40", fill="black", font=font(22))
    d.text((560, 460), "Tax (8%):    35.87", fill="black", font=font(22))
    d.text((560, 500), "TOTAL DUE:  484.27", fill="black", font=font(26))
    d.text((40, 560), "Payment due within 30 days. Thank you.", fill=(90, 90, 90), font=font(18))
    im.save(IMG / "invoice.png")

    # 2. bar chart with value labels
    im = Image.new("RGB", (900, 560), "white"); d = ImageDraw.Draw(im)
    d.text((250, 20), "Units sold by region, Q3", fill="black", font=font(28))
    data = [("North", 42, (60, 110, 200)), ("South", 67, (220, 120, 40)), ("East", 31, (60, 160, 90)),
            ("West", 58, (170, 70, 170)), ("Central", 25, (200, 60, 60))]
    d.line((90, 470, 860, 470), fill="black", width=2); d.line((90, 80, 90, 470), fill="black", width=2)
    for i, (lab, v, col) in enumerate(data):
        x = 130 + i * 145
        d.rectangle((x, 470 - v * 5, x + 90, 470), fill=col)
        d.text((x + 28, 470 - v * 5 - 30), str(v), fill="black", font=font(22))
        d.text((x + 8, 482), lab, fill="black", font=font(22))
    im.save(IMG / "chart.png")

    # 3. shapes: counting and position
    im = Image.new("RGB", (800, 600), "white"); d = ImageDraw.Draw(im)
    tri = lambda x, y, c: d.polygon([(x, y + 80), (x + 45, y), (x + 90, y + 80)], fill=c)
    tri(60, 60, (30, 140, 60)); tri(350, 260, (30, 140, 60)); tri(620, 460, (30, 140, 60))
    d.ellipse((330, 60, 420, 150), fill=(230, 170, 20)); d.ellipse((80, 440, 170, 530), fill=(230, 170, 20))
    for x, y in ((80, 250), (350, 450), (620, 250), (500, 120)):
        d.rectangle((x, y, x + 85, y + 85), fill=(40, 70, 200))
    d.ellipse((650, 50, 750, 150), fill=(200, 40, 160))   # the only purple shape, top right
    im.save(IMG / "shapes.png")

    # 4. two near-identical signs for a two-image question
    for name, price in (("sign_a.png", "$4.50"), ("sign_b.png", "$5.25")):
        im = Image.new("RGB", (640, 360), (250, 244, 225)); d = ImageDraw.Draw(im)
        d.rectangle((20, 20, 620, 340), outline=(120, 70, 20), width=6)
        d.text((150, 60), "FRESH LEMONADE", fill=(120, 70, 20), font=font(40))
        d.text((240, 160), price, fill=(180, 30, 30), font=font(64))
        d.text((215, 270), "Open 9 to 5 daily", fill=(120, 70, 20), font=font(26))
        im.save(IMG / name)

    # 5. large image with small print in one corner
    im = Image.new("RGB", (2400, 1600), (235, 240, 245)); d = ImageDraw.Draw(im)
    d.text((700, 650), "QUARTERLY REVIEW", fill=(40, 60, 90), font=font(110))
    d.text((1780, 1500), "Document ref: QR-7731-B  |  Page 4 of 12", fill=(60, 60, 60), font=font(26))
    im.save(IMG / "large.png")


# (id, image files, question, list of alternatives; each alternative is a list of strings that must all appear)
QUESTIONS = [
    ("invoice_number", ["invoice.png"], "What is the invoice number?", [["HL-20417"]]),
    ("invoice_total", ["invoice.png"], "What is the total due on this invoice?", [["484.27"]]),
    ("invoice_row", ["invoice.png"], "How many sacks of flour were ordered, and what was the amount for that line?", [["12", "222"]]),
    ("chart_tallest", ["chart.png"], "Which region sold the most units, and how many?", [["South", "67"]]),
    ("chart_lowest", ["chart.png"], "Which region sold the fewest units, and how many?", [["Central", "25"]]),
    ("chart_sum", ["chart.png"], "What is the combined total for North and West?", [["100"]]),
    ("shapes_count", ["shapes.png"], "How many triangles are in this image? Answer with a number.", [["3"], ["three"]]),
    ("shapes_squares", ["shapes.png"], "How many blue squares are in this image? Answer with a number.", [["4"], ["four"]]),
    ("shapes_corner", ["shapes.png"], "What colour is the circle in the top-right corner?", [["purple"], ["magenta"], ["pink"], ["violet"]]),
    ("two_images", ["sign_a.png", "sign_b.png"], "These two signs differ in one detail. What are the two prices?", [["4.50", "5.25"]]),
    ("large_small_print", ["large.png"], "What is the document reference printed in the bottom-right corner?", [["QR-7731-B"]]),
]


def run(build, name):
    import mlx.core as mx
    from mlx_vlm import apply_chat_template, generate, load
    model, processor = load(str(build))
    out = {"build": name, "questions": {}}
    for qid, files, q, alts in QUESTIONS:
        prompt = apply_chat_template(processor, model.config, q, num_images=len(files), enable_thinking=False)
        r = generate(model, processor, prompt, image=[str(IMG / f) for f in files], max_tokens=200, temperature=0.0)
        low = r.text.lower()
        ok = any(all(s.lower() in low for s in alt) for alt in alts)
        out["questions"][qid] = {"pass": ok, "answer": r.text.strip()[:300], "prompt_tokens": r.prompt_tokens}
        print(f"{'PASS' if ok else 'FAIL'} image {name} {qid}: {r.text.strip()[:110]!r}", flush=True)
    out["passed"] = sum(v["pass"] for v in out["questions"].values())
    out["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 1)
    (RES / f"image-{name}.json").write_text(json.dumps(out, indent=2))
    print(f"image {name}: {out['passed']} of {len(QUESTIONS)} passed", flush=True)


if __name__ == "__main__":
    make_images()
    if "--make" not in sys.argv:
        b = Path(sys.argv[1])
        run(b, sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else b.name)
