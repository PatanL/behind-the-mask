# contact sheet: python3 scripts/sheet.py out.png prefix [cols] -- labels from probe.txt
import sys, glob, os
from PIL import Image, ImageDraw, ImageFont
out, prefix = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 7
d = os.path.dirname(prefix)
labels = {}
for line in open(os.path.join(d, 'probe.txt')):
    name = line[:28].strip(); labels[name] = line[29:55].strip()
files = sorted(glob.glob(prefix + '*.png'))
ims = [Image.open(f).convert('RGB') for f in files]
w, h = ims[0].size; s = 300 / w; tw, th = int(w * s), int(h * s)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * (th + 26)), (8, 9, 14))
dr = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
except Exception: font = ImageFont.load_default()
for i, (f, im) in enumerate(zip(files, ims)):
    x, y = (i % cols) * tw, (i // cols) * (th + 26)
    sheet.paste(im.resize((tw, th)), (x, y + 26))
    dr.text((x + 6, y + 5), labels.get(os.path.basename(f)[:-4], os.path.basename(f)), fill=(200, 205, 220), font=font)
sheet.save(out); print(out, sheet.size)
